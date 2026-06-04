# Flow 2x2 Commands

Set shared paths first:

```bash
conda activate /storage/project/r-agarg35-0/eliu354/external_repos/le-wm/.conda/lewm-flow-2x2
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=$STABLEWM_HOME/experiments/flow_2x2_20260604
export PYTHONNOUSERSITE=1
```

Use `wandb.enabled=False` for local debugging only. Formal runs should keep W&B enabled.
Run `wandb login` before formal jobs. If a W&B entity is required, add `wandb.config.entity=<entity>` to the command.

## World Model Training

Original PushT:

```bash
python train.py data=pusht model=lewm seed=0 experiment.task=pusht experiment.wm_variant=original experiment.policy_variant=original output_model_name=lewm
```

Flow PushT:

```bash
python train.py data=pusht model=lewm_flow seed=0 experiment.task=pusht experiment.wm_variant=flow experiment.policy_variant=original output_model_name=lewm_flow
```

Original Cube:

```bash
python train.py data=ogb model=lewm seed=0 experiment.task=cube experiment.wm_variant=original experiment.policy_variant=original output_model_name=lewm
```

Flow Cube:

```bash
python train.py data=ogb model=lewm_flow seed=0 experiment.task=cube experiment.wm_variant=flow experiment.policy_variant=original output_model_name=lewm_flow
```

Repeat each command with `seed=1`.

## Action Flow Training

After a world model checkpoint exists under `$LEWM_EXPERIMENT_ROOT/checkpoints/{task}/{variant}/seed_{seed}/{model_name}/`, train the action-flow proposal:

```bash
python train_action_flow.py data=pusht seed=0 experiment.task=pusht experiment.wm_variant=original experiment.policy_variant=flow world_model=pusht/wm_original_policy_original/seed_0/lewm checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT
```

```bash
python train_action_flow.py data=pusht seed=0 experiment.task=pusht experiment.wm_variant=flow experiment.policy_variant=flow world_model=pusht/wm_flow_policy_original/seed_0/lewm_flow checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT
```

```bash
python train_action_flow.py data=ogb seed=0 experiment.task=cube experiment.wm_variant=original experiment.policy_variant=flow world_model=cube/wm_original_policy_original/seed_0/lewm checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT
```

```bash
python train_action_flow.py data=ogb seed=0 experiment.task=cube experiment.wm_variant=flow experiment.policy_variant=flow world_model=cube/wm_flow_policy_original/seed_0/lewm_flow checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT
```

Repeat each command with `seed=1`.

## Real-Environment Eval

Original policy/CEM:

```bash
python eval.py --config-name=pusht.yaml solver=cem seed=0 experiment.task=pusht experiment.wm_variant=original experiment.policy_variant=original policy=pusht/wm_original_policy_original/seed_0/lewm checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT
```

Flow policy proposal:

```bash
python eval.py --config-name=pusht.yaml solver=flow seed=0 experiment.task=pusht experiment.wm_variant=original experiment.policy_variant=flow policy=pusht/wm_original_policy_original/seed_0/lewm checkpoint_cache_dir=$LEWM_EXPERIMENT_ROOT solver.action_model_path=$LEWM_EXPERIMENT_ROOT/pusht/wm_original_policy_flow/seed_0/action_flow/action_flow.pt
```

For Cube, replace `--config-name=pusht.yaml` with `--config-name=cube.yaml`, `experiment.task=pusht` with `experiment.task=cube`, and checkpoint paths with `cube/...`.

## Summary

```bash
python summarize_experiments.py $LEWM_EXPERIMENT_ROOT
```
