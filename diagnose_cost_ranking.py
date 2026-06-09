import os

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from eval import get_dataset
from experiment_logging import (
    append_jsonl,
    get_run_dir,
    serializable,
    wandb_init_kwargs,
    write_run_files,
)


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def build_processors(cfg, dataset):
    processors = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        processors[col] = processor
        if col != "action":
            processors[f"goal_{col}"] = processors[col]
    return processors


def valid_eval_indices(cfg, dataset):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    step_idx = dataset.get_col_data("step_idx")
    episode_idx = dataset.get_col_data(col_name)
    lengths = []
    for ep_id in ep_indices:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    max_start_idx = np.asarray(lengths) - cfg.eval.goal_offset_steps - cfg.eval.eval_budget - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx])
    valid_mask = step_idx <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    rng = np.random.default_rng(cfg.seed)
    chosen = rng.choice(valid_indices, size=cfg.diagnostic.num_eval, replace=False)
    return np.sort(chosen)


def make_action_chunks(dataset, start_indices, cfg, processors):
    action_col = dataset.get_col_data("action")
    horizon = cfg.plan_config.horizon
    block = cfg.plan_config.action_block
    action_dim = action_col.shape[-1]
    chunks = []
    for start in start_indices:
        raw = action_col[start : start + horizon * block]
        if raw.shape[0] != horizon * block:
            raise ValueError(f"Not enough future actions at row {start}")
        norm = processors["action"].transform(raw)
        chunks.append(norm.reshape(horizon, block * action_dim))
    return torch.as_tensor(np.stack(chunks), dtype=torch.float32)


def make_info(dataset, start_indices, cfg, policy):
    goal_indices = start_indices + cfg.eval.goal_offset_steps
    start_rows = dataset.get_row_data(start_indices)
    goal_rows = dataset.get_row_data(goal_indices)

    raw = {
        "pixels": start_rows["pixels"][:, None],
        "goal": goal_rows["pixels"][:, None],
        "action": start_rows["action"][:, None],
    }
    for key in cfg.dataset.keys_to_cache:
        if key in ("pixels", "action"):
            continue
        if key in start_rows:
            raw[key] = start_rows[key][:, None]
        goal_key = f"goal_{key}"
        if key in goal_rows:
            raw[goal_key] = goal_rows[key][:, None]
    return policy._prepare_info(raw)


def expand_info_for_candidates(info, num_candidates, device):
    expanded = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            expanded[key] = value.to(device).unsqueeze(1).expand(
                value.size(0),
                num_candidates,
                *value.shape[1:],
            )
        else:
            expanded[key] = value
    return expanded


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    torch.set_grad_enabled(False)
    device = torch.device(cfg.get("device") or "cuda")
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    processors = build_processors(cfg, dataset)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    model = swm.wm.utils.load_pretrained(
        cfg.policy, cache_dir=cfg.get("checkpoint_cache_dir") or cfg.get("cache_dir")
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    policy = swm.policy.WorldModelPolicy(
        solver=lambda *_args, **_kwargs: None,
        config=swm.PlanConfig(**cfg.plan_config),
        process=processors,
        transform=transform,
    )

    start_indices = valid_eval_indices(cfg, dataset)
    info = make_info(dataset, start_indices, cfg, policy)

    expert = make_action_chunks(dataset, start_indices, cfg, processors).to(device)
    num_random = int(cfg.diagnostic.num_random)
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    random_actions = torch.randn(
        expert.shape[0],
        num_random,
        expert.shape[1],
        expert.shape[2],
        generator=generator,
        device=device,
        dtype=expert.dtype,
    )
    zero = torch.zeros_like(expert[:, None])
    candidates = torch.cat([expert[:, None], zero, random_actions], dim=1)
    info = expand_info_for_candidates(info, candidates.size(1), device)

    costs = model.get_cost(info, candidates).detach().float().cpu()
    expert_cost = costs[:, 0]
    zero_cost = costs[:, 1]
    random_costs = costs[:, 2:]
    expert_rank = (costs < expert_cost[:, None]).sum(dim=1) + 1
    random_better_frac = (random_costs < expert_cost[:, None]).float().mean(dim=1)

    record = {
        "stage": "cost_ranking_diag",
        "policy": cfg.policy,
        "seed": int(cfg.seed),
        "start_indices": start_indices.tolist(),
        "num_eval": int(cfg.diagnostic.num_eval),
        "num_random": num_random,
        "mean_expert_cost": float(expert_cost.mean()),
        "mean_zero_cost": float(zero_cost.mean()),
        "mean_random_cost": float(random_costs.mean()),
        "mean_expert_rank": float(expert_rank.float().mean()),
        "mean_random_better_frac": float(random_better_frac.mean()),
        "expert_costs": expert_cost.tolist(),
        "zero_costs": zero_cost.tolist(),
        "expert_ranks": expert_rank.tolist(),
        "random_better_frac": random_better_frac.tolist(),
    }

    run_dir = get_run_dir(cfg, swm, "eval")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_files(
        run_dir,
        cfg,
        "cost_ranking_diag",
        extra={
            "policy": cfg.policy,
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "checkpoint_cache_dir": cfg.get("checkpoint_cache_dir") or cfg.get("cache_dir"),
        },
    )
    append_jsonl(run_dir / "metrics.jsonl", record)

    wb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        wb_run = wandb.init(
            **wandb_init_kwargs(cfg.wandb.config),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wb_run.log(
            {
                "diag/mean_expert_cost": record["mean_expert_cost"],
                "diag/mean_zero_cost": record["mean_zero_cost"],
                "diag/mean_random_cost": record["mean_random_cost"],
                "diag/mean_expert_rank": record["mean_expert_rank"],
                "diag/mean_random_better_frac": record["mean_random_better_frac"],
            }
        )
        wb_run.finish()

    print(serializable(record))


if __name__ == "__main__":
    run()
