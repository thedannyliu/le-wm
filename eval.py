import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from experiment_logging import (
    append_jsonl,
    get_run_dir,
    serializable,
    wandb_init_kwargs,
    write_run_files,
)

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    if hasattr(swm.data, "HDF5Dataset"):
        return swm.data.HDF5Dataset(
            dataset_name,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=dataset_path,
        )

    candidates = [Path(dataset_name)]
    if not str(dataset_name).endswith(".h5"):
        candidates.append(Path(f"{dataset_name}.h5"))
    candidates.extend([dataset_path / candidate for candidate in list(candidates)])
    dataset_ref = next((path for path in candidates if path.exists()), dataset_name)
    return swm.data.load_dataset(
        str(dataset_ref),
        cache_dir=str(dataset_path),
        keys_to_cache=cfg.dataset.keys_to_cache,
    )

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy_name = cfg.get("policy", "random")

    run_dir = get_run_dir(cfg, swm, "eval")
    video_dir = run_dir / "videos"
    results_file = run_dir / cfg.output.filename
    run_dir.mkdir(parents=True, exist_ok=True)

    wb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        wb_run = wandb.init(
            **wandb_init_kwargs(cfg.wandb.config),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    if policy_name != "random":
        checkpoint_cache_dir = cfg.get("checkpoint_cache_dir") or cfg.get("cache_dir")
        model = swm.wm.utils.load_pretrained(policy_name, cache_dir=checkpoint_cache_dir)
        device = cfg.get("device") or cfg.solver.get("device", "cuda")
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    write_run_files(
        run_dir,
        cfg,
        "eval",
        extra={
            "policy": policy_name,
            "checkpoint_cache_dir": cfg.get("checkpoint_cache_dir") or cfg.get("cache_dir"),
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "results_file": str(results_file),
            "video_dir": str(video_dir),
            "wandb": OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
        },
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_dir,
    )
    end_time = time.time()
    
    print(metrics)

    eval_record = {
        "stage": "real_env_eval",
        "metrics": metrics,
        "evaluation_time": end_time - start_time,
        "seed": cfg.seed,
        "policy": policy_name,
        "solver": OmegaConf.to_container(cfg.get("solver", {}), resolve=True),
        "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
        "video_dir": str(video_dir),
    }
    if policy_name != "random" and hasattr(policy, "solver") and hasattr(policy.solver, "last_solve_stats"):
        eval_record["solver_stats"] = policy.solver.last_solve_stats
    append_jsonl(run_dir / "metrics.jsonl", eval_record)

    if wb_run is not None:
        wandb_metrics = {
            "eval/evaluation_time": end_time - start_time,
            "eval/seed": cfg.seed,
        }
        if isinstance(metrics, dict):
            for key, value in serializable(metrics).items():
                if isinstance(value, (int, float, bool)):
                    wandb_metrics[f"eval/{key}"] = value
        if "solver_stats" in eval_record:
            for key, value in serializable(eval_record["solver_stats"]).items():
                if isinstance(value, (int, float, bool)):
                    wandb_metrics[f"eval/{key}"] = value
        wb_run.log(wandb_metrics)
        for video_path in sorted(video_dir.glob("*.mp4")):
            wb_run.log({f"eval/video/{video_path.stem}": wandb.Video(str(video_path))})
        wb_run.finish()

    with results_file.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
