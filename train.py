import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from experiment_logging import (
    JsonlMetricsCallback,
    get_experiment_root,
    get_run_dir,
    get_task_name,
    get_variant_name,
    write_run_files,
)
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def find_latest_last_ckpt(run_dir):
    checkpoints = sorted(
        (run_dir / "spt" / "runs").glob("**/checkpoints/last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    is_flow_predictor = hasattr(self.model.predictor, "flow_loss")

    # LeWM loss
    if is_flow_predictor:
        output["flow_loss"] = self.model.predictor.flow_loss(ctx_emb, ctx_act, tgt_emb)
        flow_pred_weight = cfg.loss.get("flow_pred", {}).get("weight", 0.0)
        pred_context = torch.enable_grad() if flow_pred_weight else torch.no_grad()
        with pred_context:
            pred_emb = self.model.predict(ctx_emb, ctx_act)
            output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        if flow_pred_weight:
            output["endpoint_loss"] = output["pred_loss"]
    else:
        pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    prediction_objective = output.get("flow_loss", output["pred_loss"])
    if is_flow_predictor:
        prediction_objective = prediction_objective + cfg.loss.get("flow_pred", {}).get("weight", 0.0) * output["pred_loss"]
    output["loss"] = prediction_objective + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = cfg.get("cache_dir") or os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    experiment_run_dir = get_run_dir(cfg, swm, "train")
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    if cfg.get("experiment", {}).get("use_run_dir_for_lightning", True):
        run_dir = experiment_run_dir / "lightning"
    experiment_root = get_experiment_root(cfg, swm)
    spt_cache_dir = cfg.get("experiment", {}).get("spt_cache_dir") or str(
        experiment_run_dir / "spt"
    )
    spt.set(cache_dir=spt_cache_dir)
    checkpoint_run_name = cfg.get("checkpoint_run_name") or str(
        Path(get_task_name(cfg), get_variant_name(cfg), f"seed_{cfg.seed}", cfg.output_model_name)
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    resume_cfg = cfg.get("resume", {})
    ckpt_path = resume_cfg.get("ckpt_path")
    ckpt_path = Path(ckpt_path).expanduser() if ckpt_path else None
    if ckpt_path is None and resume_cfg.get("auto", False):
        ckpt_path = find_latest_last_ckpt(experiment_run_dir)
    if ckpt_path is None:
        legacy_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
        ckpt_path = legacy_ckpt_path if legacy_ckpt_path.exists() else None
    resume_weights_only = False if ckpt_path else True

    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    write_run_files(
        experiment_run_dir,
        cfg,
        "train",
        extra={
            "checkpoint_dir": str(run_dir),
            "checkpoint_cache_dir": str(experiment_root),
            "checkpoint_run_name": checkpoint_run_name,
            "spt_cache_dir": spt_cache_dir,
            "output_model_name": cfg.output_model_name,
            "metrics_path": str(experiment_run_dir / "metrics.jsonl"),
            "resume_ckpt_path": str(ckpt_path) if ckpt_path else None,
            "resume_weights_only": resume_weights_only,
            "wandb": OmegaConf.to_container(cfg.wandb, resolve=True),
        },
    )

    object_dump_callback = SaveCkptCallback(
        run_name=checkpoint_run_name,
        cfg=cfg.model,
        epoch_interval=1,
        cache_dir=experiment_root,
    )

    callbacks = [
        object_dump_callback,
        JsonlMetricsCallback(
            experiment_run_dir / "metrics.jsonl",
            every_n_steps=cfg.get("logging", {}).get("jsonl_every_n_steps", 50),
        ),
    ]
    step_ckpt_every = cfg.get("logging", {}).get("requeue_checkpoint_every_n_steps", 0)
    if step_ckpt_every:
        callbacks.append(
            ModelCheckpoint(
                filename="last",
                save_top_k=-1,
                every_n_train_steps=int(step_ckpt_every),
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
        )
    if cfg.wandb.enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path and ckpt_path.exists() else None,
        weights_only=resume_weights_only,
    )

    manager()
    return


if __name__ == "__main__":
    run()
