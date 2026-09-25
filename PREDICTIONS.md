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
