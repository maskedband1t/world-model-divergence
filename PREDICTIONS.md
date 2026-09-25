# Pre-registration

**Written 2026-09-24, before any run. Not edited after results land.**
Corrections and amendments go in a dated block at the bottom, never by rewriting a line above.

---

## The question

A policy that plans inside a world model acts on frames the world model invented.
Those frames drift from real dynamics. The standard defence is a **fixed imagination
horizon**: stop dreaming at step H, where H was chosen in advance by a person.

That is a frozen rule, and it is the same object this programme measured in the
human–robot seat: a hand-written threshold standing in for a judgment nobody measured.

Two questions, in plain words:

1. **Does the world model know when its dream stopped being true?**
2. **Is the fixed horizon leaving value on the table — in both directions?**
   (Dreaming past the truth, and stopping short of it.)

## The setup

**Substrate.** DIAMOND (Alonso et al., NeurIPS 2024 Spotlight), pretrained Atari 100k
world model + policy, weights from the Hugging Face Hub. MIT licensed.
Chosen because **Atari is deterministic given seed and action sequence**, so the real
emulator gives exact ground truth for where a rollout went wrong. No other open
world model at this scale gives that for free.

**Games (pre-committed, 4).** `Breakout`, `Pong`, `MsPacman`, `Boxing`.
Chosen for different dynamics: sparse rigid-body, adversarial, multi-object clutter,
dense contact. `Pong` is included as an expected-weak case — see Prior work below.

**Protocol.** From a real state, step the world model H frames and the real emulator
H frames under the **same action sequence**. Record per-step distance between matched
frames. Repeat across seeds.

## Defining "diverged" — and the noise floor first

Pixel distance drifts even when the dynamics are right, so a raw threshold would
manufacture a result. Before any arm is compared:

**Control run (run this first, report it first).** Two *real* emulator rollouts from
the same state under the same actions, differing only by ALE's own stochasticity
(sticky actions). The per-step distance between them is the **noise floor** — the
distance two trajectories show while both are still true.

**Divergence step** := the first step at which the world-model-vs-real distance
exceeds the 95th percentile of that noise floor at the same step index.

Distance metric: **LPIPS**, with raw **L2** reported alongside as the dumb metric.
If the two metrics disagree on any headline claim, that disagreement is the finding
and both get reported.

## The arms

| Arm | What it is | Role |
|---|---|---|
| **Fixed horizon** | DIAMOND's default, H = 50 | The frozen rule. The dumb baseline. |
| **Calibrated gate** | Stop when the model's own confidence drops below a threshold | The thing under test |
| **Real emulator** | Ground truth | The oracle / ceiling |

**Confidence signal.** Sample N = 8 stochastic rollouts from identical conditioning
(the diffusion sampler is stochastic; see `config/trainer.yaml`,
`world_model_env.diffusion_sampler`). Spread across those samples at step k is the
model's own stated uncertainty. No new training, no new head.

**Factorial, not confounded.** game (4) × arm (3) × seed (≥10). Every cell run.
Horizon swept, not fixed at one value, so the comparison is a curve and not a point.

## Predictions

Numbered, falsifiable, dated above. Confidence in brackets is my own before seeing data.

- **P1 — The default horizon is already past the truth in at least one game.** [0.75]
  Median divergence step < 50 for at least one of the four. If true, the shipped
  default is dreaming into fiction and nobody measured it.

- **P2 — Divergence step varies more across games than across seeds within a game.** [0.7]
  Between-game spread > 2× within-game seed spread. If false, a per-game fixed horizon
  is as good as anything adaptive and the whole premise weakens.

- **P3 — Sample spread carries signal about divergence.** [0.6]
  AUROC > 0.65 for predicting "diverged by step k", pooled across games.

- **P4 — But it is not calibrated out of the box.** [0.8]
  ECE > 0.15 when spread is mapped to a probability of still being faithful.
  This is the RLCD result's shape: ranking survives, the number does not.

- **P5 — A calibrated gate keeps more faithful frames than fixed-H at equal budget.** [0.5]
  At matched *average* generated frames per rollout, the gated arm yields more
  frames inside the noise floor than H = 50. Genuinely uncertain; this is the payoff
  claim and the one most likely to fail.

- **P6 — Pong is the weak case.** [0.65]
  Pong shows the earliest divergence step of the four, consistent with prior work
  reporting a weak Pong world model behind a strong Pong agent.

## What would make me wrong

Stated in advance so it cannot be explained away afterwards:

- **P3 fails at chance (AUROC ≈ 0.5).** Most likely reading: diffusion sample spread
  tracks pixel-level texture noise, not dynamics divergence. That is a real negative
  and it ships. It would say the model's own sampler carries no usable trust signal
  and an external probe is required.
- **P1 fails in all four games.** H = 50 is conservative everywhere; the fixed rule is
  fine; the interesting question moves to whether it is *too* conservative.
- **The noise floor swallows the effect.** If world-model-vs-real distance sits inside
  the two-real-rollouts band for most of the horizon, there is no divergence to detect
  at this scale and the result is "the instrument needs a harder substrate."

**Pre-committed:** every outcome above ships, including all-negative. A negative with a
measured noise floor is a result. The misses stay in the record next to the hits.

## Out of scope, and why

- **No policy retraining.** Downstream return under a gated world model needs a full
  RL retrain. Not reachable in the time budget. The metric here is faithful frames at
  equal budget, which is measurable without it. Retraining is future work, stated as such.
- **No MIRA leg.** MIRA released code and the Rocket Science dataset but **no pretrained
  weights**, and the codec needs external DINOv3 weights. Training it is out of budget.
  The same instrument on Rocket Science — which ships game state alongside every frame,
  and so offers the same ground truth at real scale — is the stated next step.
- **No claim about latent-space MBRL.** See Prior work.

## Prior work this does not duplicate

Rollout uncertainty in model-based RL is an active lane, not an empty one:

- *Uncertainty-Aware Robotic World Model* (RWM-U), ICLR 2026 — epistemic uncertainty in
  **latent/state space**, ensemble variance as a MOPO-style penalty, offline MBRL on robots.
- *Horizon-Calibrated Uncertainty* (HAUWM), ICLR 2026 — ensemble of dynamics heads,
  variance forced to grow monotonically with horizon.
- *Improving Weak World Models Behind Strong Agents in Atari Pong* (arXiv 2607.15142) —
  source of P6.

What is not covered there, and is what this measures: an **interactive generative video**
world model rather than a latent dynamics model; **strict calibration** (ECE, a probability
that matches its hit rate) rather than variance-as-penalty; and the **gate** framing, where
the number is spent by a governor rather than folded into a loss.

## Amendments

*(dated entries only; nothing above is edited)*

### 2026-09-24 — amendment 1: the noise floor needs sticky actions turned on

Found while reading DIAMOND's source before the first run; nothing above is edited.

**What the pre-registration got wrong.** It said the control run would be "two *real*
emulator rollouts ... differing only by ALE's own stochasticity (sticky actions)."
DIAMOND trains and evaluates on `*NoFrameskip-v4`, and **v4 has no sticky actions**.
Two real rollouts from the same seed under the same action sequence are bit-identical,
so the floor as written would have been exactly zero and every divergence threshold
built on it would have been meaningless.

**The correction.** The control run explicitly sets
`repeat_action_probability = 0.25` — the ALE standard (Machado et al. 2018) — for
**both** real rollouts. The floor then answers the question it was always meant to
answer: how far apart do two physically plausible realisations of the same action
sequence sit while **both are still true**?

**What this buys, which the original design did not have.** The divergence run keeps
sticky actions **off**. The real env is then deterministic, so the world-model-vs-real
distance contains **no environment noise at all** — every unit of it is model error.
Attribution is unambiguous, which is stronger than what was pre-registered.

**What it costs, stated plainly.** The floor is measured under a slightly different
env setting (sticky on) than the divergence it calibrates (sticky off). The floor is
therefore an upper bound on tolerable distance rather than a matched control. Any
headline claim that is sensitive to that gap must say so. No prediction P1–P6 changes.

### 2026-09-24 — amendment 2: where the confidence signal actually comes from

DIAMOND's shipped sampler config sets `s_churn: 0.0` and `order: 1`. The reverse
diffusion is therefore deterministic given its starting latent, and the **only** source
of variation between two rollouts from identical conditioning is the initial
`torch.randn` draw.

P3 and P4 are unchanged, but the mechanism is now specific: sample spread measures how
much the outcome depends on the initial latent. If P3 fails at chance, the reading in
"What would make me wrong" stands and sharpens — the initial latent would be washing out
rather than carrying information about dynamics uncertainty, and an external probe or a
non-zero `s_churn` sweep becomes the follow-up.

### 2026-09-24 — amendment 3: the control run had a confound, found before any run

The original control compared a **closed-loop** arm (policy acting, reacting to sticky
deviations) against an **open-loop** arm (replaying the first arm's actions). Those two
differ by the policy's ability to react as well as by the sticky noise under test, so
the floor would have been inflated by an effect that has nothing to do with "how far
apart two true realisations sit."

**Corrected design.** A pilot rollout (policy acting, sticky **off**) produces one action
sequence. Both control arms replay that sequence **open-loop** with sticky **on**, at
different seeds. They now differ only by the ALE's sticky draws.

**Side benefit that narrows amendment 1's stated cost.** The pilot uses the same seed as
the divergence run, so the floor and the divergence it calibrates are measured on the
**same action sequence**. Only the sticky setting still differs.

### 2026-09-24 — amendment 4: the analysis is frozen before the data exists

`harness/analysis.py` was written and committed before a single run had been executed.
Every number reported in the README is produced by it: the floor percentile, the
divergence step, AUROC, held-out ECE, and the equal-budget gate comparison. Freezing it
first is what makes the predictions above falsifiable rather than decorative — the
analysis cannot be reshaped once the data is in view.

It ships with a self-test on synthetic data, including a **negative control**: a spread
signal that is pure noise must return AUROC ≈ .5. It returns .507. The pipeline does not
manufacture a positive from noise.

**A limitation recorded now, not after seeing results.** LPIPS with AlexNet is applied to
64x64 frames, where its deepest features are only a few pixels across. It is used because
it is the standard perceptual metric, not because it is well suited to this resolution.
Raw L2 is reported beside it in every table for exactly this reason, and a disagreement
between them is a finding and not a nuisance to be resolved in favour of whichever is
more convenient.

### 2026-09-24 — amendment 5: the most likely way P3 fails, named in advance

P3 predicts that sample spread carries signal about divergence. The reading already given
for its failure is that spread tracks texture noise. A second and more interesting failure
is now named before it can be discovered and rationalised:

**The samples may be confidently wrong in the same way.** Once a rollout has left the real
trajectory, every sample can agree closely with every other sample while all of them agree
with reality not at all — low spread, high divergence. Sample agreement measures agreement
among samples, which is only a proxy for truth while the model is still tracking.

If that is what the data shows, it is the world-model analogue of the owned copy in the
sibling repo: blind off its own distribution and increasingly sure about it. That would be
a result, not a failed experiment, and it would argue that a trust signal has to come from
somewhere other than the model's own agreement with itself.

### 2026-09-24 — amendment 6: a CPU pilot falsifies the premise of P3–P5

Run before any GPU time, on CPU, to check the instrument had dynamic range. It does not.
Reported here because the pilot changes the experiment, and a design changed after seeing
data must say so in the open.

**Pilot 1 — sample spread does not exist.** Four rollouts from identical conditioning,
Breakout, t0 = 20, 8 steps. Mean sample spread **0.0002–0.0004** against a mean error
versus the real emulator of **0.0073**: the signal is 2–6 % of the quantity it is meant
to predict. Sweeping `s_churn` from 0.0 to 2.0 and denoising steps from 3 to 10 moves
spread from .00023 to at most .00043. DIAMOND's denoiser collapses whatever latent it is
given, so the sampler is effectively deterministic and **P3 and P4 have no signal to
measure**.

**Pilot 2 — no cheap internal signal beats a step counter.** Three seeds x 25 steps
(n = 75), Breakout, correlation with per-step error against the real emulator:

| signal | corr with error | p10 -> p90 |
|---|---|---|
| denoiser residual (last denoising step) | -0.178 | .00280 -> .00330 |
| end-head entropy | -0.019 | 0 -> 0 |
| reward-head entropy | 0.259 | 0 -> 0 |
| model's own frame-to-frame delta | 0.110 | .0208 -> .0606 |
| **step index** | **0.458** | 3 -> 23 |

Both `rew_end_model` heads are saturated at zero entropy over this horizon on this game —
expected, since Breakout rewards and terminations are rare here, but it means they carry
no usable signal in this regime. Every internal signal tested is flat, and **the single
best predictor of how wrong the model is, is how many steps it has dreamed.**

**What this does to the claim.** A fixed horizon *is* a step counter. If nothing beats a
step counter, the field's frozen constant is not a strawman — it is close to the best
available rule, and the README's framing is wrong in a way that matters.

**Caveats, so this is not over-read.** One game, n = 75, 25 steps, three seeds, no noise
floor applied yet. Correlation with raw error is not the pre-registered quantity, which is
the divergence step relative to the floor. Error grows with step, so *any* monotone
function of step correlates somewhat. This is a feasibility check on the instrument, not
a test of P1–P6.

**Consequence, recorded before the next run.** P1, P2 and P6 are unaffected and stand as
written. P3, P4 and P5 are retained unchanged so their failure is on the record, and the
experiment gains a fourth arm and a harder baseline:

> **The step counter is promoted from "the thing we are beating" to the dumb baseline any
> signal must beat.** A signal that predicts divergence no better than "how long have I
> been dreaming" is worthless, because that is what H already encodes. Arms become:
> step counter (baseline) · internal signals (pilot says flat) · an external probe trained
> on conditioning features · oracle.

The external probe is the fallback named in amendment 5 and is now the load-bearing arm
rather than a contingency.
