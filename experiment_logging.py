import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


DEFAULT_EXPERIMENT_NAME = "flow_2x2_20260604"


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_cache_root(swm_module):
    return Path(swm_module.data.utils.get_cache_dir())


def get_experiment_root(cfg, swm_module):
    configured = cfg.get("experiment", {}).get("output_root")
    if configured:
        return Path(configured).expanduser()
    env_root = os.environ.get("LEWM_EXPERIMENT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return get_cache_root(swm_module) / "experiments" / DEFAULT_EXPERIMENT_NAME


def get_variant_name(cfg):
    exp = cfg.get("experiment", {})
    return exp.get(
        "variant",
        f"wm_{exp.get('wm_variant', 'original')}_policy_{exp.get('policy_variant', 'original')}",
    )


def get_task_name(cfg):
    exp = cfg.get("experiment", {})
    if exp.get("task"):
        return exp.task
    if cfg.get("eval", {}).get("dataset_name"):
        return str(cfg.eval.dataset_name).replace("/", "_")
    if cfg.get("data", {}).get("dataset", {}).get("name"):
        return str(cfg.data.dataset.name).replace("/", "_").replace(".h5", "")
    return "unknown_task"


def get_run_dir(cfg, swm_module, phase):
    exp = cfg.get("experiment", {})
    configured = exp.get("run_dir")
    if configured:
        return Path(configured).expanduser()
    return (
        get_experiment_root(cfg, swm_module)
        / get_task_name(cfg)
        / get_variant_name(cfg)
        / f"seed_{cfg.get('seed', 'unknown')}"
        / phase
    )


def serializable(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def scalar_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        value = serializable(value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(serializable(record), sort_keys=True) + "\n")


def write_run_files(run_dir, cfg, phase, extra=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (run_dir / "git_sha.txt").write_text(get_git_sha() + "\n")

    manifest = {
        "phase": phase,
        "created_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": get_git_sha(),
        "command": sys.argv,
        "run_dir": str(run_dir),
    }
    if extra:
        manifest.update(serializable(extra))
    OmegaConf.save(OmegaConf.create(manifest), run_dir / "run_manifest.yaml")
    return manifest


def wandb_init_kwargs(wandb_config):
    try:
        kwargs = OmegaConf.to_container(wandb_config, resolve=True)
    except Exception:
        kwargs = OmegaConf.to_container(wandb_config, resolve=False)
    kwargs.pop("log_model", None)
    if "save_dir" in kwargs and "dir" not in kwargs:
        kwargs["dir"] = kwargs.pop("save_dir")
    return {
        key: value
        for key, value in kwargs.items()
        if not (isinstance(value, str) and "${" in value)
    }


class JsonlMetricsCallback(Callback):
    def __init__(self, metrics_path, every_n_steps=50):
        self.metrics_path = Path(metrics_path)
        self.every_n_steps = every_n_steps

    def _record(self, trainer, stage):
        metrics = scalar_metrics(trainer.callback_metrics)
        metrics.update(
            {
                "stage": stage,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "wall_time": time.time(),
            }
        )
        append_jsonl(self.metrics_path, metrics)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        if trainer.global_step % self.every_n_steps == 0:
            self._record(trainer, "train_step")

    def on_train_epoch_end(self, trainer, pl_module):
        self._record(trainer, "train_epoch")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._record(trainer, "val_epoch")
