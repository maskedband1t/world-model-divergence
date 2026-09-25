# The external probe arm

Written 2026-09-24, after the amendment 6 pilot showed no internal signal beats a step
counter, and before any probe has been trained. Predictions are in PREDICTIONS.md (P7–P10).

## What changed, and why it is an improvement

The original design pitted a calibrated signal against the field's fixed horizon H = 50.
The pilot exposed the flaw: **a fixed horizon is a step counter**, and a step counter was
the best predictor of error we could find. Beating H = 50 might only mean "50 was the wrong
number," which says nothing about whether a calibrated judgment is worth anything.

So the comparison is decomposed into two questions that the old design confounded:

| gap | what it isolates | prediction |
|---|---|---|
| fixed H = 50 → **best step counter** | the cost of shipping an untuned constant | P1, P7 |
| best step counter → **probe** | whether looking at the content helps **at all** | P8, P9 |

The second gap is the real question, and nothing in the original design measured it.

## The three arms

**Arm A — the clock.** `P(faithful | k)` fitted on the step index alone. Because this is
monotone decreasing in k, thresholding it is *exactly* a fixed horizon with a per-game
tuned H. Arm A is therefore **the best possible fixed horizon**, not a strawman.

**Arm B — the probe.** `P(faithful | k, content)` — the clock plus features of what the
model is actually generating. Strictly nests Arm A, so `B − A` is the value of content
with the clock already paid for.

**Oracle.** The real emulator. Ceiling.

Every arm is compared at **equal average generated-frame budget**, as in P5.

## What the probe may see

Only what a deployed system has at step k: its own generated frames and the actions taken.
Never the real frames, never the future, never the label.

| feature | cost | why |
|---|---|---|
| step index `k` | free | the clock; also in Arm A |
| **policy entropy** over `logits_act` | 1 forward pass | the actor-critic was trained on *real* frames. When the dream drifts off-manifold, the agent should stop recognising the state. A *different model* judging the generator. |
| **critic value** estimate | same pass | as above: a scalar that should behave oddly on nonsense |
| actor-critic encoder embedding, pooled | same pass | learned frame representation, free — no feature engineering |
| frame-to-frame delta magnitude | free | pilot corr 0.110: weak but non-zero |
| rolling std of deltas, window 4 | free | temporal instability rather than instantaneous change |
| pixel mean / std | free | the dumb features; if these win, the learned ones were not needed |

The interesting bet is policy entropy and critic value: **use a model trained on reality to
audit a model generating fiction.** Nothing inside the generator carried signal
(amendment 6); the auditor is a different network.

Cost: one actor-critic forward pass per generated frame, negligible beside diffusion
sampling. **The probe arm adds essentially no GPU time** — same rollouts, more logging.

## Head, fitting, and honesty controls

- **Head:** L2-regularised logistic regression, plain numpy. Deliberately the simplest
  thing that can combine features. If a calibrated probability needs something cleverer,
  that is itself the finding.
- **Cross-fitting by seed.** The head never scores a rollout from a seed it trained on.
  Arms A and B use identical folds.
- **Calibration** measured held-out, by ECE, as everywhere else in this repo.

**The permutation control, which is what makes any claim here credible.** Shuffle the
labels **within each step index**. That destroys every content signal while leaving the
clock perfectly intact. Under this shuffle:

- Arm A must be unchanged (it only ever saw `k`).
- Arm B must collapse to Arm A.

If Arm B keeps an advantage under the permutation, that advantage was the clock leaking
through the fit, and the result is void. This control runs before any headline number.

## What a negative looks like, and why it still ships

If `B − A` is within noise, the finding is stated plainly: **on a diffusion world model at
this scale, the content of a rollout carries no usable trust signal beyond how long it has
been running, so a tuned fixed horizon is the right engineering answer.** That is a real
result about a real system, it is the same shape as the sibling repo's "rules win where
they were written," and it is what the pilot currently predicts.

The gap still worth reporting in that case is `H = 50 → Arm A`: the price the field pays
for an untuned constant, measured per game.
