# LeWM Flow-WM Progress Report

Generated: 2026-06-11 EDT

## Storyline

We tried to replace LeWM's deterministic latent world model with a flow-based world model while keeping the official PushT pipeline fixed: same data, encoder, action embedder, checkpointing, W&B logging, CEM planner, and real-environment eval.

The result is currently negative but useful: **native LeWM remains much stronger, and the flow-WM failures point to objective/planner mismatch rather than a simple training bug.** Residual flow-WM can reduce flow-matching loss, but CEM needs a reliable deterministic multi-step cost surface. Endpoint loss fixes one-step prediction enough to produce some real-env signal, but success plateaus around `20-30%`, far below native LeWM. The next research direction should be planner-cost alignment, not lower flow loss.

## Key Results

Success is real-environment PushT success rate. `medium10` means 10 episodes; `full50` means 50 episodes. `Pred loss` is validation next-latent MSE; lower is better.

| Setting | Eval | Epochs s0/s1/s2 | Pred loss | Success (%) | Main read |
| --- | --- | --- | ---: | ---: | --- |
| Native LeWM + CEM, selected | full50 | 48 / 83 / 75 | `0.0019` | `87.3 +/- 1.2` | Baseline to beat. |
| Native LeWM + CEM, final | full50 | 100 / 100 / 100 | `0.1053` | `58.0 +/- 45.1` | Final epoch is misleading because seed 0 collapses. |
| Native LeWM + action-flow, final | full50 | 100 / 100 / 100 | same WM | `44.0 +/- 33.3` | Not a fair action-flow test; uses collapsed final WMs. |
| Native LeWM + CEM, early screen | medium10 | 35 / 28 / 33 | `0.0028` | `90.0 +/- 10.0` | Native is already strong before 100 epochs. |
| Residual flow-WM + CEM | medium10 | 16 / 13 / 17 | `1.2075` | `0.0 +/- 0.0` | Flow loss alone fails for CEM. |
| Endpoint flow-WM + CEM, first screen | medium10 | 14 / 12 / 12 | `0.0083` | `30.0 +/- 17.3` | Endpoint loss repairs part of the failure. |
| Endpoint flow-WM + CEM, later screen | medium10 | 30 / 28 / 28 | `0.0055` | `23.3 +/- 5.8` | Lower pred loss does not improve control. |
| Endpoint flow-WM + CEM, latest screen | medium10 | 34 / 32 / 33 | `0.0053` | `20.0 +/- 10.0` | Plateau confirmed. |

One single-seed full50 check was run for endpoint flow-WM: seed 0 epoch 14 achieved `30%`, while native seed 0 epoch 48 achieved `88%`.

## Compute Comparison

Training time is logged wall-time from the first validation epoch to the evaluated checkpoint, averaged over seeds. It is useful for order-of-magnitude comparison but is not a clean hardware benchmark because jobs resume under `embers`. Inference time is `eval/evaluation_time / episodes`.

| Setting | Train time to checkpoint (h) | Inference time (s/episode) | Eval success (%) | Compute read |
| --- | ---: | ---: | ---: | --- |
| Native LeWM + CEM, selected full50 | `61.3` | `1.31` | `87.3` | Best accuracy and fastest CEM. |
| Native LeWM + CEM, early medium10 | `31.1` | `1.67` | `90.0` | Strong before full training. |
| Native LeWM + action-flow, final full50 | n/a | `0.44` | `44.0` | Fast policy-side sampler, but unfair checkpoint. |
| Residual flow-WM + CEM | `16.6` | `9.98` | `0.0` | Slower inference and no success. |
| Endpoint flow-WM + CEM, first screen | `16.7` | `5.47` | `30.0` | Better than residual, still slow and weak. |
| Endpoint flow-WM + CEM, later screen | `40.7` | `5.67` | `23.3` | More training does not help. |
| Endpoint flow-WM + CEM, latest screen | `46.3` | `7.42` | `20.0` | Worse compute/quality tradeoff than native. |

Compute signal: current flow-WM is not Pareto-efficient. It is slower at CEM inference and substantially less successful. If flow remains useful, it likely needs to be used differently, for example as a planner-aware model or as an action proposal module with selected native checkpoints.

## Strong Training Signals

These are the signals that most clearly decide what to try next.

| Signal | Metric evidence | What it tells us |
| --- | --- | --- |
| Checkpoint selection is mandatory | Native seed 0: pred loss `0.00239 -> 0.31281`, full50 `88% -> 6%` from epoch 48 to 100. | Do not report final checkpoint only; select by validation/eval signals. |
| Flow loss alone is misaligned | Residual flow-WM: flow loss `0.0612`, but pred loss `1.2075`, medium10 `0%`, cost rank `56.75-65.56`. | A flow-matching objective can train while giving CEM an unusable deterministic rollout. |
| Endpoint loss fixes one-step prediction | Endpoint first screen pred loss `0.0083` vs residual `1.2075`; medium10 improves `0% -> 30%`. | The diagnosis was right: CEM needs endpoint-aligned predictions. |
| Endpoint loss is not sufficient | Endpoint pred loss improves `0.0083 -> 0.0055 -> 0.0053`, but success goes `30.0% -> 23.3% -> 20.0%`. | Stop optimizing only one-step endpoint MSE; target planner-cost alignment. |
| Cost ranking is necessary but insufficient | Endpoint seed 1 rank `1.19`, random-better frac `0.0015`, but medium10 only `20%`. | Offline expert-vs-random cost ranking is helpful, but not enough for closed-loop robustness. |
| Flow-WM has poor compute tradeoff | Native CEM `1.31s/ep` at `87.3%`; endpoint latest `7.42s/ep` at `20.0%`; residual `9.98s/ep` at `0%`. | Flow-WM needs a stronger reason to pay its inference cost. |

## Pipeline Difference

Only the predictor/objective was intended to change in the world-model comparison.

| Variant | World model | Objective | Planner/eval | Status |
| --- | --- | --- | --- | --- |
| Native LeWM | deterministic ARPredictor | next-latent pred loss + SIGReg | original CEM | strong baseline |
| Residual flow-WM | conditional residual flow | flow matching + SIGReg | original CEM | fails |
| Endpoint flow-WM | residual flow + endpoint prediction | flow + endpoint/pred loss + SIGReg | original CEM | partial repair, plateau |
| Action-flow proposal | native WM unchanged | action-sequence flow imitation | learned proposal/eval | needs fair selected-checkpoint eval |

## Problems & Next Steps

1. **Stop spending large eval budget on unchanged endpoint-flow if latest cost-rank does not reveal a new signal.**
   Latest medium10 already shows a flat trend at `20-30%`.

2. **Make the next WM objective planner-aware.**
   Add a loss that directly compares LeWM planner cost for expert chunks vs random/CEM candidate chunks, e.g. a margin/ranking loss.

3. **Test multi-sample or stochastic flow rollout only if it changes the cost surface.**
   Endpoint-flow compresses the model to one deterministic endpoint; if uncertainty is useful, CEM needs to see it through rollout costs.

4. **Re-run action-flow on selected native checkpoints.**
   The existing action-flow result is fast (`0.44s/episode`) but confounded by epoch-100 WM collapse.

5. **Keep native selected full50 as the fairness anchor.**
   Current target: native LeWM + CEM selected checkpoints, `87.3 +/- 1.2%` full50 at `1.31s/episode`.

## Current Pending Item

Latest3 endpoint medium10 finished at `30/10/20%`. Latest3 endpoint cost-ranking jobs are still pending:

| Job | Seed | Epoch | Purpose |
| ---: | ---: | ---: | --- |
| `9820449` | 0 | 34 | cost-rank |
| `9820450` | 1 | 32 | cost-rank |
| `9820451` | 2 | 33 | cost-rank |
