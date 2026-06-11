# LeWM Flow-WM Progress Report

Generated: 2026-06-11 EDT

Sources:

- Local outputs: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/`
- W&B project: `lewm-flow-2x2`
- Key summary table: `docs/assets/lewm_flow_progress_20260611/lewm_story_key_results.csv`
- Cost-rank summary table: `docs/assets/lewm_flow_progress_20260611/lewm_cost_rank_summary.csv`

## Conclusion & Insights

We are testing whether flow can improve LeWM by replacing the learned world-model dynamics while keeping the original PushT data, training loop, checkpointing, CEM planner, and real-environment eval pipeline fixed.

**Current answer: direct flow-WM replacement is not promising under the original LeWM/CEM interface.**

- Native LeWM + CEM remains the anchor: `87.3 +/- 1.2%` full50 success at `1.31s/episode`.
- Residual flow-WM is a clear failure mode: low flow loss (`0.0612`) but bad deterministic prediction (`1.2075`) and `0%` medium10 success.
- Endpoint flow-WM repairs one-step prediction (`0.0083 -> 0.0055 -> 0.0053`) but not control (`30.0% -> 23.3% -> 20.0%`).
- Runtime moves the same direction as performance moves against us: endpoint flow-WM is `~5.7x` slower than native selected CEM (`7.42s` vs `1.31s/episode`) while much less successful.

**Main insight:** the bottleneck is likely not model capacity or insufficient training time. The signal points to an interface mismatch: flow objectives can improve distributional or endpoint prediction, but CEM needs a deterministic, multi-step, action-conditional cost surface that ranks candidate action sequences correctly under closed-loop rollouts.

**Research decision from current evidence:** do not keep scaling the same endpoint-flow WM as the main path. The next useful experiments should change the training signal or planner interface: planner-aware cost ranking/margin losses, multi-step rollout consistency on CEM-like action chunks, or uncertainty-aware CEM if we actually use flow samples during planning. A separate action-flow proposal is still worth testing because it can use flow where sampling is naturally useful while keeping the reliable native WM.

## Key Results

`Pred loss` is validation next-latent MSE. `Train h` is logged wall-time from the first validation epoch to the evaluated checkpoint, averaged over seeds; it is not a clean hardware benchmark because jobs resume under `embers`.

Fair comparison boundary: all claims below are PushT-only and use the same LeWM data/eval pipeline. Full50 is the main metric. Medium10 is a faster screening metric; it is still informative here because the flow-WM gap is large, consistent across seeds, and supported by validation/cost diagnostics.

| Setting | Eval | Epochs s0/s1/s2 | Pred loss | Success (%) | Train h | Sec/episode | Signal |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Native LeWM + CEM, selected | full50 | 48 / 83 / 75 | `0.0019` | `87.3 +/- 1.2` | `61.3` | `1.31` | Fair baseline anchor. |
| Native LeWM + CEM, final | full50 | 100 / 100 / 100 | `0.1053` | `58.0 +/- 45.1` | `85.8` | `1.43` | Checkpoint selection matters. |
| Native LeWM + action-flow, final | full50 | 100 / 100 / 100 | same WM | `44.0 +/- 33.3` | n/a | `0.44` | Fast but checkpoint-confounded. |
| Native LeWM + CEM, early | medium10 | 35 / 28 / 33 | `0.0028` | `90.0 +/- 10.0` | `31.1` | `1.67` | Native learns useful planner cost early. |
| Residual flow-WM + CEM | medium10 | 16 / 13 / 17 | `1.2075` | `0.0 +/- 0.0` | `16.6` | `9.98` | Flow loss does not expose a CEM-usable point prediction. |
| Endpoint flow-WM + CEM, first | medium10 | 14 / 12 / 12 | `0.0083` | `30.0 +/- 17.3` | `16.7` | `5.47` | Endpoint target fixes only the first failure. |
| Endpoint flow-WM + CEM, later | medium10 | 30 / 28 / 28 | `0.0055` | `23.3 +/- 5.8` | `40.7` | `5.67` | Better MSE does not become better control. |
| Endpoint flow-WM + CEM, latest | medium10 | 34 / 32 / 33 | `0.0053` | `20.0 +/- 10.0` | `46.3` | `7.42` | Plateau; not an undertrained-model story. |

-> The useful result is causal direction, not just ranking: native LeWM succeeds when validation prediction and planner cost ranking agree; flow-WM can improve its own losses without producing a robust CEM cost surface.

## Pipeline (w/ Diff)

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

-> This isolates the failure to the WM/planner interface. Residual flow fails before control because CEM needs a point rollout. Endpoint flow makes that point rollout numerically reasonable, but the planner still does not get a reliable action-sequence cost landscape.

![Success vs inference cost](assets/lewm_flow_progress_20260611/lewm_success_vs_inference_cost.png)

### 2. Endpoint-Flow Training Ablation

| Endpoint eval set | Epochs s0/s1/s2 | Pred loss | Flow loss | Success (%) |
| --- | --- | ---: | ---: | ---: |
| First medium10 | 14 / 12 / 12 | `0.0083` | `0.0614` | `30.0` |
| Later medium10 | 30 / 28 / 28 | `0.0055` | `0.0526` | `23.3` |
| Latest medium10 | 34 / 32 / 33 | `0.0053` | `0.0513` | `20.0` |

-> This is the strongest "do not just train longer" signal. The model becomes better at the supervised endpoint target while closed-loop behavior gets no better, so the missing target is not more endpoint MSE; it is planner-relevant multi-step cost calibration.

![Endpoint-flow trend](assets/lewm_flow_progress_20260611/lewm_endpoint_trend.png)

### 3. Prediction Loss vs Control

| Comparison | Metric signal | Control signal | Research implication |
| --- | --- | --- | --- |
| Native selected vs final seed 0 | pred loss `0.00239 -> 0.31281` | full50 `88% -> 6%` | validation pred loss is meaningful for native checkpoint selection. |
| Residual flow-WM | flow loss `0.0612`, pred loss `1.2075` | medium10 `0%` | flow loss alone is misleading. |
| Endpoint flow-WM | pred loss `0.0083 -> 0.0053` | medium10 `30% -> 20%` | one-step endpoint MSE is not enough. |

-> Prediction loss is useful only when it matches the planner interface. For native LeWM it does; for flow-WM it becomes a weak proxy. The next metric should measure whether the WM ranks CEM candidate rollouts the same way the real environment would, not only whether the next latent is close.

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

-> Cost ranking separates native/residual cleanly, but it does not fully explain endpoint-flow. Endpoint checkpoints can rank expert chunks above random chunks while still failing in real env. That means the diagnostic is too easy: expert-vs-random single-step ranking is necessary, but CEM fails on harder near-miss candidates and compounding multi-step errors.

![Endpoint cost-rank trend](assets/lewm_flow_progress_20260611/lewm_endpoint_cost_rank_trend.png)

### 5. Compute / Runtime Ablation

| Setting | Train h | Sec/episode | Success (%) | Compute read |
| --- | ---: | ---: | ---: | --- |
| Native selected CEM | `61.3` | `1.31` | `87.3` | best Pareto point |
| Native action-flow final | n/a | `0.44` | `44.0` | fast, but checkpoint-confounded |
| Residual flow-WM | `16.6` | `9.98` | `0.0` | dominated |
| Endpoint flow-WM latest | `46.3` | `7.42` | `20.0` | dominated |

-> Runtime is a research constraint, not just engineering overhead. Since CEM calls the WM many times per control step, a flow-WM must either deliver a large success gain or expose useful uncertainty to the planner. The current version does neither, so it is Pareto-dominated.

## Problems & Next Steps

Where the project is currently stuck:

1. **The planner consumes a deterministic cost surface, but flow-WM is trained as a distributional transition model.**
   This is why residual flow can lower flow loss while giving CEM a bad rollout.

2. **Endpoint loss solves the visible prediction bug but not the decision problem.**
   Endpoint-flow reaches low next-latent MSE, yet success plateaus. The missing supervision is probably on action-sequence ranking and multi-step compounding error, not single-step endpoint accuracy.

3. **The diagnostic is not hard enough yet.**
   Expert-vs-random cost ranking catches residual-flow failure, but endpoint-flow can pass this test and still fail in real env. We need CEM-candidate ranking, near-miss negatives, or rollout-level calibration.

4. **Flow-WM currently loses on compute.**
   It is slower and less successful, so the only reason to keep it is if we change the planner/objective enough for flow's distributional structure to matter.

Next experiments that are actually informative:

| Direction | Why it follows from the evidence | Success criterion |
| --- | --- | --- |
| Planner-aware WM loss | Endpoint MSE improved without control gain. | Expert/CEM-good chunks receive lower predicted cost than near-miss CEM candidates. |
| Multi-step latent rollout loss | CEM evaluates action sequences, not isolated next latents. | Cost rank and real-env success improve together over 5-step chunks. |
| Uncertainty-aware flow planning | Flow is only useful if sampled uncertainty changes CEM's decision. | Multi-sample or risk-sensitive CEM improves success enough to justify runtime. |
| Action-flow proposal on selected native WM | Flow may be better as an action sampler than as the dynamics model. | Faster eval than native CEM without using collapsed epoch-100 WMs. |
| Stop unchanged endpoint-flow scaling | Latest training already shows lower pred loss with no success gain. | Only continue if a new objective/planner change is added. |

Advisor-level decision point: should we make flow compatible with the original deterministic CEM planner, or change the planner to consume flow uncertainty? Current evidence favors the first as the next controlled experiment, because it preserves the original LeWM pipeline and tests the smallest hypothesis change.

## Appendix

Output roots:

- Native LeWM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605`
- Residual flow-WM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608`
- Endpoint flow-WM: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609`
- Cost diagnostics: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609`

Caveat:

- Latest3 cost-rank jobs wrote local metrics under the default diagnostic variant path (`wm_flow_endpoint_policy_original/seed_x/eval`) because the diagnostic sbatch does not yet pass `experiment.variant`; W&B summaries and Slurm logs match the values reported above.
