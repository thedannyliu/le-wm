# LeWM Flow-WM Progress Report

Generated: 2026-06-11 EDT

Sources:

- Local outputs: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/`
- W&B project: `lewm-flow-2x2`
- Key summary table: `docs/assets/lewm_flow_progress_20260611/lewm_story_key_results.csv`
- Cost-rank summary table: `docs/assets/lewm_flow_progress_20260611/lewm_cost_rank_summary.csv`

## Conclusion & Insights

- **Native LeWM is still the anchor.**
  - Selected native LeWM + CEM reaches `87.3 +/- 1.2%` full50 success at `1.31s/episode`.
  - Endpoint flow-WM is far lower: best completed full50 is seed 0 epoch 14 at `30%`; latest medium10 is `20.0 +/- 10.0%`.

- **The flow-WM failure is an objective/planner mismatch, not just an unfinished run.**
  - Residual flow-WM reaches low flow loss (`0.0612`) but has very bad deterministic pred loss (`1.2075`) and `0%` medium10 success.
  - CEM needs a stable deterministic multi-step cost surface; standard flow matching does not give that by itself.

- **Endpoint loss fixes the obvious one-step prediction problem, but not closed-loop control.**
  - Endpoint pred loss improves from `0.0083 -> 0.0055 -> 0.0053`.
  - Success does not improve: `30.0% -> 23.3% -> 20.0%`.
  - This is the strongest signal that the next objective should be planner-cost aligned, not just lower endpoint MSE.

- **Runtime is part of the scientific result.**
  - Native CEM: `1.31s/episode` at `87.3%`.
  - Endpoint flow-WM: `7.42s/episode` at `20.0%`.
  - Residual flow-WM: `9.98s/episode` at `0%`.
  - Current flow-WM is not Pareto-efficient.

## Key Results

`Pred loss` is validation next-latent MSE. `Train h` is logged wall-time from the first validation epoch to the evaluated checkpoint, averaged over seeds; it is not a clean hardware benchmark because jobs resume under `embers`.

| Setting | Eval | Epochs s0/s1/s2 | Pred loss | Success (%) | Train h | Sec/episode | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Native LeWM + CEM, selected | full50 | 48 / 83 / 75 | `0.0019` | `87.3 +/- 1.2` | `61.3` | `1.31` | Best baseline. |
| Native LeWM + CEM, final | full50 | 100 / 100 / 100 | `0.1053` | `58.0 +/- 45.1` | `85.8` | `1.43` | Final epoch is misleading. |
| Native LeWM + action-flow, final | full50 | 100 / 100 / 100 | same WM | `44.0 +/- 33.3` | n/a | `0.44` | Fast, but unfair collapsed checkpoint. |
| Native LeWM + CEM, early | medium10 | 35 / 28 / 33 | `0.0028` | `90.0 +/- 10.0` | `31.1` | `1.67` | Strong before 100 epochs. |
| Residual flow-WM + CEM | medium10 | 16 / 13 / 17 | `1.2075` | `0.0 +/- 0.0` | `16.6` | `9.98` | Flow loss alone fails. |
| Endpoint flow-WM + CEM, first | medium10 | 14 / 12 / 12 | `0.0083` | `30.0 +/- 17.3` | `16.7` | `5.47` | Partial repair. |
| Endpoint flow-WM + CEM, later | medium10 | 30 / 28 / 28 | `0.0055` | `23.3 +/- 5.8` | `40.7` | `5.67` | Lower pred loss, no gain. |
| Endpoint flow-WM + CEM, latest | medium10 | 34 / 32 / 33 | `0.0053` | `20.0 +/- 10.0` | `46.3` | `7.42` | Plateau confirmed. |

-> The key result is not just "flow-WM is worse"; it is worse while being slower, and the training metrics explain why.

## LeWM Pipeline

Algorithm: LeWM PushT Training and Evaluation Pipeline (`->` marks our modification)

Input:
    PushT expert dataset: `pusht_expert_train.lance`.
        -> Current report focuses on PushT only; OGBench/Cube is not used for the claims here.

    Each trajectory provides:
        - RGB pixels
        - action
        - proprio
        - state

    Training sample construction:
        - history size: 3 frames
        - prediction target: 1 next latent
        - frameskip / action block: 5
        - train split: 90%

Model:
    LeWM is a latent world model trained from expert demonstrations.

    Shared modules kept fixed across WM variants:
        - ViT-tiny image encoder: pixels -> latent features
        - projector MLP: encoder features -> latent z
        - action Embedder: action chunk -> action embedding
        - prediction projector MLP
        - SIGReg regularization

    Native world model:
        - ARPredictor: latent history z, action embedding -> next latent z
        - objective: next-latent prediction loss + SIGReg

    Flow world-model variants:
        - Residual flow-WM:
            ARPredictor -> ConditionalFlowPredictor over latent residual dynamics
            objective -> flow-matching loss + SIGReg
        - Endpoint flow-WM:
            ConditionalFlowPredictor + endpoint prediction target
            objective -> flow loss + endpoint/pred loss + SIGReg

    Policy / planner:
        - default action selector: CEM planner using the learned world model
        - CEM config: 300 samples, 30 optimization steps, top-30 elites
        - planning horizon: 5, receding horizon: 5
        - optional action-flow proposal:
            CEM proposal sampler -> learned conditional action-flow proposal
            native WM is unchanged in this ablation

Training Pipeline:
    Step 1: Load expert demonstrations
        Load PushT expert trajectories from `pusht_expert_train.lance`.
        Normalize/cache action, proprio, and state statistics.

    Step 2: Train native LeWM baseline
        Update:
            - ViT encoder
            - projector / prediction projector
            - action Embedder
            - ARPredictor

        Optimize:
            next-latent prediction loss + SIGReg

    Step 3: Train flow-WM ablations
        Keep the same data, encoder, projector, action Embedder, optimizer,
        dataloader, checkpointing, W&B logging, and eval pipeline.

        Change only:
            native ARPredictor -> flow-based predictor

        Residual flow-WM:
            optimize flow-matching loss.

        Endpoint flow-WM:
            optimize flow-matching loss plus endpoint/prediction loss.

    Step 4: Checkpoint selection
        Record validation metrics:
            - pred loss
            - flow loss, for flow-WM
            - endpoint loss, for endpoint flow-WM
            - SIGReg loss

        Important:
            final epoch is not safe; native seed 0 drops from `88%` to `6%`.

Evaluation Pipeline:
    Main eval: closed-loop PushT real-environment evaluation.

    For each checkpoint:
        1. Reset PushT eval environment.
        2. Set eval state and goal from dataset-defined starts.
        3. At each planning step:
            - observe pixels / state
            - encode pixels into latent z
            - run CEM with the learned world model
            - score candidate action chunks by predicted latent goal cost
            - execute the selected action block
            - replan with new environment feedback
        4. Record success rate, per-episode successes, evaluation time, videos, and W&B metrics.

    Eval protocols:
        - full50: 50 real-environment episodes; main downstream metric
        - medium10: 10 real-environment episodes; checkpoint-screening metric

    Diagnostics:
        - Cost ranking:
            expert action chunks should score below random chunks under the LeWM planner cost.
        - Action-flow eval:
            CEM proposal sampler -> learned flow proposal, native WM unchanged.

## Ablations & Insights

### 1. World-Model Architecture Ablation

| WM | Objective | Eval | Pred loss | Success (%) | Sec/episode | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Native ARPredictor | pred + SIGReg | full50 | `0.0019` | `87.3` | `1.31` | Strong and cheap. |
| Residual flow-WM | flow + SIGReg | medium10 | `1.2075` | `0.0` | `9.98` | Flow loss not usable by CEM. |
| Endpoint flow-WM | flow + endpoint + SIGReg | medium10 | `0.0053` | `20.0` | `7.42` | One-step repair, closed-loop plateau. |

-> Replacing the deterministic WM with flow dynamics is not working under the current CEM interface.

![Success vs inference cost](assets/lewm_flow_progress_20260611/lewm_success_vs_inference_cost.png)

### 2. Endpoint-Flow Training Ablation

| Endpoint eval set | Epochs s0/s1/s2 | Pred loss | Flow loss | Success (%) |
| --- | --- | ---: | ---: | ---: |
| First medium10 | 14 / 12 / 12 | `0.0083` | `0.0614` | `30.0` |
| Later medium10 | 30 / 28 / 28 | `0.0055` | `0.0526` | `23.3` |
| Latest medium10 | 34 / 32 / 33 | `0.0053` | `0.0513` | `20.0` |

-> Endpoint prediction keeps improving, but real-env success and planner cost ranking do not improve.

![Endpoint-flow trend](assets/lewm_flow_progress_20260611/lewm_endpoint_trend.png)

### 3. Prediction Loss vs Control

| Comparison | Metric signal | Control signal | Research implication |
| --- | --- | --- | --- |
| Native selected vs final seed 0 | pred loss `0.00239 -> 0.31281` | full50 `88% -> 6%` | validation pred loss is meaningful for native checkpoint selection. |
| Residual flow-WM | flow loss `0.0612`, pred loss `1.2075` | medium10 `0%` | flow loss alone is misleading. |
| Endpoint flow-WM | pred loss `0.0083 -> 0.0053` | medium10 `30% -> 20%` | one-step endpoint MSE is not enough. |

-> The next metric should measure planner-relevant multi-step cost quality, not only one-step prediction.

![Pred loss vs success](assets/lewm_flow_progress_20260611/lewm_pred_loss_vs_success.png)

### 4. Cost-Ranking Diagnostic

Lower expert rank is better. Rank `1` means expert action chunks are scored best among random alternatives.

| Model / checkpoint | Mean expert rank | Random-better frac | Real-env read |
| --- | ---: | ---: | --- |
| Native seed 2 e75 | `1.00` | `0.0000` | full50 `88%` |
| Residual flow seed 2 e17 | `56.75` | `0.4302` | medium10 `0%` |
| Residual flow seed 2 e24 | `65.56` | `0.5015` | near-random cost surface |
| Endpoint early mean, e11/e9/e9 | `6.48` | `0.0422` | early quick eval `2/9` |
| Endpoint mid mean, e19/e17/e17 | `5.77` | `0.0368` | medium10 `20.0%` |
| Endpoint latest2 mean | `5.17` | `0.0322` | medium10 `23.3%` |
| Endpoint latest3 mean | `6.77` | `0.0449` | medium10 `20.0%` |

-> Cost ranking is a useful diagnostic, but endpoint-flow still needs a better closed-loop planner objective.

![Endpoint cost-rank trend](assets/lewm_flow_progress_20260611/lewm_endpoint_cost_rank_trend.png)

### 5. Compute / Runtime Ablation

| Setting | Train h | Sec/episode | Success (%) | Compute read |
| --- | ---: | ---: | ---: | --- |
| Native selected CEM | `61.3` | `1.31` | `87.3` | best Pareto point |
| Native action-flow final | n/a | `0.44` | `44.0` | fast, but checkpoint-confounded |
| Residual flow-WM | `16.6` | `9.98` | `0.0` | dominated |
| Endpoint flow-WM latest | `46.3` | `7.42` | `20.0` | dominated |

-> Runtime should be treated as part of the result because CEM repeatedly queries the WM during control.

## Problems & Next Steps

Problems:

- Flow-WM has a planner/objective mismatch.
  - Flow matching can optimize while CEM receives an unusable deterministic rollout cost.
- Endpoint flow-WM fixes one-step prediction but not closed-loop robustness.
- Current flow-WM is slower than native CEM while much less successful.
- Action-flow is not fairly tested yet because current runs use epoch-100 native WMs, including a collapsed seed.

Next:

- Add planner-aware WM objectives:
  - expert-vs-random cost margin loss
  - CEM-candidate ranking loss
  - multi-step latent rollout loss under candidate action chunks
- Test stochastic or multi-sample flow rollout only if CEM uses the uncertainty in its cost.
- Re-run action-flow proposal on selected native checkpoints.
- Keep native selected full50 as the fairness anchor: `87.3 +/- 1.2%`, `1.31s/episode`.

## Appendix

Output roots:

- Native LeWM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605`
- Residual flow-WM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608`
- Endpoint flow-WM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609`
- Cost diagnostics: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609`

Caveat:

- Latest3 cost-rank jobs wrote local metrics under the default diagnostic variant path (`wm_flow_endpoint_policy_original/seed_x/eval`) because the diagnostic sbatch does not yet pass `experiment.variant`; W&B summaries and Slurm logs match the values reported above.
