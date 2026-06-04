from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict


class FlowProposalSolver:
    """Sample action chunks from a flow policy, then score with a world model."""

    def __init__(
        self,
        model,
        action_model: Any = None,
        action_model_path: str | None = None,
        action_model_cache_dir: str | None = None,
        condition_dim: int = 384,
        num_samples: int = 300,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        include_zero_action: bool = True,
    ):
        self.model = model
        self.action_model_cfg = action_model
        self.action_model = action_model if isinstance(action_model, torch.nn.Module) else None
        self.action_model_path = action_model_path
        self.action_model_cache_dir = action_model_cache_dir
        self.condition_dim = condition_dim
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.include_zero_action = include_zero_action
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)
        self.last_solve_stats = {}
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logger.warning(
                f"Action space is discrete, got {type(action_space)}. FlowProposalSolver expects continuous actions."
            )

        self._maybe_build_action_model()

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _maybe_build_action_model(self):
        if self.action_model is None and self.action_model_cfg is not None:
            cfg = self.action_model_cfg
            if isinstance(cfg, (DictConfig, dict)):
                cfg = OmegaConf.create(cfg)
                with open_dict(cfg):
                    cfg.condition_dim = self.condition_dim
                    cfg.action_dim = self.action_dim
                    cfg.horizon = self.horizon
                self.action_model = hydra.utils.instantiate(cfg)

        if self.action_model is None:
            return

        if self.action_model_path:
            checkpoint_path = Path(self.action_model_path)
            if self.action_model_cache_dir and not checkpoint_path.is_absolute():
                checkpoint_path = Path(self.action_model_cache_dir) / checkpoint_path
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.action_model.load_state_dict(state_dict)

        self.action_model = self.action_model.to(self.device)
        self.action_model.eval()
        self.action_model.requires_grad_(False)

    def _pad_init_action(self, init_action, total_envs):
        if init_action is None:
            return torch.zeros(
                total_envs, self.horizon, self.action_dim, device=self.device, dtype=self.dtype
            )
        init_action = init_action.to(device=self.device, dtype=self.dtype)
        remaining = self.horizon - init_action.shape[1]
        if remaining > 0:
            pad = torch.zeros(
                total_envs, remaining, self.action_dim, device=self.device, dtype=self.dtype
            )
            init_action = torch.cat([init_action, pad], dim=1)
        return init_action[:, : self.horizon]

    def _tensor_batch_len(self, info_dict):
        for value in info_dict.values():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return len(value)
        raise ValueError("info_dict has no batched tensor or ndarray values")

    def _to_model_device(self, value):
        if torch.is_tensor(value):
            dtype = self.dtype if value.is_floating_point() else None
            return value.to(device=self.device, dtype=dtype)
        return value

    def _encode_last(self, info):
        info = {k: self._to_model_device(v) for k, v in info.items() if torch.is_tensor(v)}
        encoded = self.model.encode(info)
        return encoded["emb"][:, -1]

    def _build_condition(self, info_dict):
        current = {
            k: v
            for k, v in info_dict.items()
            if torch.is_tensor(v) and not k.startswith("goal_")
        }
        current.pop("goal", None)
        current.pop("action", None)
        current_emb = self._encode_last(current)

        goal = {k: v for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("goal", None)
        goal.pop("action", None)
        goal_emb = self._encode_last(goal)
        return torch.cat([current_emb, goal_emb], dim=-1)

    def _expand_info(self, info_dict, total_envs):
        expanded_infos = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                target_dtype = self.dtype if v.is_floating_point() else None
                v = v.to(device=self.device, dtype=target_dtype)
                expanded_infos[k] = v.unsqueeze(1).expand(
                    total_envs, self.num_samples, *v.shape[1:]
                )
            elif isinstance(v, np.ndarray):
                expanded_infos[k] = np.repeat(v[:, None, ...], self.num_samples, axis=1)
            else:
                expanded_infos[k] = v
        return expanded_infos

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        total_envs = self._tensor_batch_len(info_dict)
        init_action = self._pad_init_action(init_action, total_envs)

        if self.action_model is not None:
            condition = self._build_condition(info_dict)
            candidates = self.action_model.sample(condition, num_samples=self.num_samples)
            candidates = candidates.to(device=self.device, dtype=self.dtype)
        else:
            candidates = torch.randn(
                total_envs,
                self.num_samples,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )

        if self.include_zero_action:
            candidates[:, 0] = init_action

        expanded_infos = self._expand_info(info_dict, total_envs)
        costs = self.model.get_cost(expanded_infos, candidates)
        best_idx = torch.argmin(costs, dim=1)
        batch_idx = torch.arange(total_envs, device=self.device)
        actions = candidates[batch_idx, best_idx]

        self.last_solve_stats = {
            "num_samples": self.num_samples,
            "wm_rollouts": total_envs * self.num_samples,
            "mean_best_cost": costs[batch_idx, best_idx].float().mean().item(),
        }

        return {
            "actions": actions.detach().cpu(),
            "costs": costs[batch_idx, best_idx].detach().cpu().tolist(),
            "all_costs": costs.detach().cpu(),
            "best_idx": best_idx.detach().cpu(),
            "flow_stats": self.last_solve_stats,
        }
