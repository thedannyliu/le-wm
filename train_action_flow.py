import os
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import wandb
from omegaconf import OmegaConf, open_dict

from experiment_logging import append_jsonl, get_run_dir, wandb_init_kwargs, write_run_files
from utils import get_column_normalizer, get_img_preprocessor


def encode_condition(world_model, batch):
    current = {"pixels": batch["pixels"][:, :1]}
    goal = {"pixels": batch["pixels"][:, -1:]}
    current_emb = world_model.encode(current)["emb"][:, -1]
    goal_emb = world_model.encode(goal)["emb"][:, -1]
    return torch.cat([current_emb, goal_emb], dim=-1)


def move_batch(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def run_epoch(model, world_model, loader, optimizer, cfg, stage, run_dir, wb_run):
    is_train = stage == "train"
    model.train(is_train)
    losses = []
    for step, batch in enumerate(loader):
        batch = move_batch(batch, cfg.device)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        target = batch["action"][:, : cfg.plan_config.horizon]
        with torch.no_grad():
            condition = encode_condition(world_model, batch)
        loss = model.flow_loss(condition, target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("gradient_clip_val"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            optimizer.step()

        loss_value = loss.detach().float().item()
        losses.append(loss_value)
        record = {
            "stage": f"{stage}_step",
            "step": step,
            "action_flow_loss": loss_value,
        }
        append_jsonl(run_dir / "metrics.jsonl", record)
        if wb_run is not None and (step % cfg.logging.wandb_every_n_steps == 0):
            wb_run.log({f"{stage}/action_flow_loss": loss_value})

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    record = {
        "stage": f"{stage}_epoch",
        "action_flow_loss": mean_loss,
    }
    append_jsonl(run_dir / "metrics.jsonl", record)
    if wb_run is not None:
        wb_run.log({f"{stage}/action_flow_loss_epoch": mean_loss})
    return mean_loss


@hydra.main(version_base=None, config_path="./config/train", config_name="action_flow")
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = cfg.get("cache_dir") or os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.action_model.action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.action_model.horizon = cfg.plan_config.horizon
        cfg.action_model.condition_dim = 2 * cfg.embed_dim

    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    checkpoint_cache_dir = cfg.get("checkpoint_cache_dir") or cfg.get("cache_dir")
    world_model = swm.wm.utils.load_pretrained(cfg.world_model, cache_dir=checkpoint_cache_dir)
    world_model = world_model.to(cfg.device).eval()
    world_model.requires_grad_(False)
    world_model.interpolate_pos_encoding = True

    action_model = hydra.utils.instantiate(cfg.action_model).to(cfg.device)
    optimizer = torch.optim.AdamW(action_model.parameters(), **cfg.optimizer)

    run_dir = get_run_dir(cfg, swm, "action_flow")
    write_run_files(
        run_dir,
        cfg,
        "action_flow",
        extra={
            "world_model": cfg.world_model,
            "checkpoint_cache_dir": checkpoint_cache_dir,
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "action_model_path": str(run_dir / "action_flow.pt"),
        },
    )

    wb_run = None
    if cfg.wandb.enabled:
        wb_run = wandb.init(
            **wandb_init_kwargs(cfg.wandb.config),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_val = float("inf")
    for epoch in range(cfg.trainer.max_epochs):
        train_loss = run_epoch(action_model, world_model, train, optimizer, cfg, "train", run_dir, wb_run)
        val_loss = run_epoch(action_model, world_model, val, optimizer, cfg, "val", run_dir, wb_run)
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "stage": "epoch",
                "epoch": epoch,
                "train_action_flow_loss": train_loss,
                "val_action_flow_loss": val_loss,
            },
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(action_model.state_dict(), run_dir / "action_flow.pt")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    run()
