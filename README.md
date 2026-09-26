# When to Stop Dreaming

**How far into a rollout can you trust a world model — and does the model itself know?**

Anurag Akkiraju · September 2026 · *Calibrated decisions at the policy–world-model
boundary. Same instrument as [the human–robot
boundary](https://github.com/maskedband1t/RLCD), one seat over.*

> **STATUS: in progress.** Pre-registration is complete and dated
> ([PREDICTIONS.md](PREDICTIONS.md)); runs are underway. Every number below marked
> ‹TBD› is a placeholder and no claim is made until it is filled from a logged run.

**The gap.** A policy that plans inside a world model acts on frames the world model
invented, and invented frames drift from real dynamics. The standard defence is a
**fixed imagination horizon** — stop dreaming at step H, where H was picked in advance
by a person. Nobody measures whether H is right. It is a hand-written threshold standing
in for a judgment that was never checked.

**The bet.** That the model's **own confidence**, scored so the probability matches its
hit rate, is worth more in that seat than the frozen number — and that the frozen number
is wrong in both directions: dreaming past the truth in some games, stopping short of it
in others.

**Measured.** ‹TBD› runs across four Atari games × three arms × ‹TBD› seeds, every
prediction written and dated before its run, against the fixed horizon the field ships
and against the real emulator as ground truth. The misses stay in the record.

**One claim, two seats.** A uniform constant applied to a non-uniform stream is wasteful at one end
and dangerous at the other, and the field's fix is always a hand-written detector for which regime it
is in: a fixed horizon deciding how long a policy may dream, a rule program deciding when a robot asks
for help, a gripper-event rule deciding which motion may be accelerated. This programme asks whether a
**calibrated per-step number is the general form of that detector** — and measures what that is
worth against the right control. Seat 1 has now answered part of it: a state-dependent test buys
**+0.100 handled events** over an arm that skips at the *same rate* with no test at all, which
buys **+0.000**, replicated on two banks. So the detector's *content* earns its place; skipping
alone does not.

**What the same work did not support, stated here rather than buried.** The original form of this
claim was that a calibrated number is general *because it transfers where a hand-written rule does
not*. On that bench it does not survive: a raw, uncalibrated threshold transferred nearly as well
for **behaviour** (conditional accuracy drifting under .08 across a 53-point base-rate shift), and
an attempt to show physical thresholds transfer worse produced an interval spanning zero. What
calibration demonstrably buys there is that the number attached to the gate is roughly **true**
(half the calibration error off its fit distribution) — which matters for reasoning about the gate
and composing it with other numbers, not for where it fires.

Measured here at the policy–world-model
boundary; measured at the human–robot boundary in
[RLCD](https://github.com/maskedband1t/RLCD).

**This is not hypothetical.** [Argon Robotics](https://www.argonrobotics.ai/research/scaling-and-speeding-up-robots-in-the-real-world)
ship a deployed skip gate tuned to 2 cm in clean scenes and 0.3 cm in human-occupied ones — one threshold
per context, switched by hand — and an adaptive speed-up whose optimum sits at 4× for reasons they state
plainly that they do not know. Three published, load-bearing, unexplained constants, counting DIAMOND's
H = 50. The frozen threshold is endemic, and practitioners say so in their own limitations sections.

**A prediction about someone else's result, recorded before this repo has any of its own:** Argon's 4×
optimum is set by their segmenter's error rate, not by the policy's tolerance for speed — so a calibrated
segmenter would move it. We have no access to their rig, so this stands as a stated, falsifiable
consequence of the claim rather than a measured finding.

---

## Why this can be measured at all

**Atari is deterministic given a seed and an action sequence.** Step the world model H
frames and step the real emulator H frames under the *same* actions, and you know
exactly where the dream stopped being true. No other open world model at this scale
hands you ground truth for free.

The substrate is [DIAMOND](https://github.com/eloialonso/diamond) (Alonso, Jelley,
Kanervisto, Micheli et al., NeurIPS 2024 Spotlight) — pretrained Atari 100k world model
and policy, weights on the Hugging Face Hub, MIT licensed. No training is required to
ask the question, which is why it fits in a weekend.

## The noise floor, before anything else

Pixel distance drifts even when the dynamics are right. A raw threshold would
manufacture a result, so the first run is a control and it is reported first:

> Two **real** rollouts from the same state under the same actions, differing only by
> the emulator's own stochasticity. The distance between them is how far apart two
> trajectories sit **while both are still true**.

A rollout counts as **diverged** at the first step where world-model-vs-real distance
exceeds the 95th percentile of that floor at the same step index. Everything downstream
is measured against that band.

Noise floor, measured: ‹TBD›

## The three arms

| Arm | What it is | Role |
|---|---|---|
| **Fixed horizon** (H = 50) | DIAMOND's shipped default | The frozen rule. The dumb baseline. |
| **Calibrated gate** | Stop when the model's own confidence falls below a threshold | The thing under test |
| **Real emulator** | Ground truth | The oracle. The ceiling. |

The confidence signal is free: the diffusion sampler is stochastic, so eight rollouts
from identical conditioning give a spread, and that spread is the model's own stated
uncertainty. No new head, no new training.

## At a glance

| the question | what we found |
|---|---|
| Is the shipped default horizon past the point where the dream stopped being true? | ‹TBD› |
| Does the model's own sample spread predict the divergence step? | ‹TBD› (AUROC) |
| Is that spread **calibrated**, or does it only rank? | ‹TBD› (ECE) |
| At equal generated-frame budget, does a calibrated gate keep more faithful frames than H = 50? | ‹TBD› |
| Which game breaks first? | ‹TBD› |

## The headline figure

‹TBD — reliability on the states the rollout itself creates: stated confidence against
realised faithfulness, gate against fixed horizon, with the noise-floor band drawn.›

*Deliberately the same figure as the sibling repo's reliability plot. Same instrument,
different boundary: there the states came from a robot's own actions, here from a world
model's own rollout.*

## Same instrument, second seat

This is one programme, not two projects. The measurement is identical and only the
boundary moves:

| Human–Robot Boundary | Policy–World-Model Boundary |
|---|---|
| the situations its own **actions** create | the frames its own **rollout** creates |
| a frozen **rule program** | a fixed **imagination horizon** |
| an oracle that reads true state | the real emulator |
| pre-registered banks: written vs unwritten | pre-registered seeds: inside vs beyond horizon |
| ECE on self-induced states | ECE on self-induced states |

In the robot seat the finding was that ranking survives distribution shift and the
number does not — a strong open model kept its accuracy while its stated sureness drifted
.135 on the states its own actions created, where a calibrated model held within .02.
This asks whether a world model's confidence has the same disease.

## Honest placement

Rollout uncertainty in model-based RL is an active lane, not an empty one.
[RWM-U](https://arxiv.org/abs/2504.16680) and HAUWM (both ICLR 2026) estimate epistemic
uncertainty in **latent/state space** and fold it into a loss as an ensemble-variance
penalty. What is measured here is different in three ways, and only in three ways:

1. an **interactive generative video** world model, not a latent dynamics model;
2. **strict calibration** — a probability that matches its hit rate, scored by ECE —
   not variance-as-penalty;
3. the **gate** framing, where the number is spent by a governor that can stop, rather
   than absorbed into training.

If those three turn out not to matter, that is also a result.

## What this does not do

- **No policy retraining.** Downstream return under a gated world model needs a full RL
  retrain, which is outside this budget. The metric here is faithful frames at equal
  budget. Retraining is future work and is named as such.
- **No MIRA leg, yet.** [MIRA](https://github.com/mira-wm/mira) released training code
  and the [Rocket Science](https://huggingface.co/datasets/kyutai/rocket-science)
  dataset under Apache 2.0, but **no pretrained weights**. Rocket Science ships game
  state (ball, cars, score) alongside every frame, which is the same ground-truth
  affordance that makes this measurement possible — at real scale, on a multiplayer
  generative video model. That is the next seat.
- **No claim beyond Atari.** Four games is four games.

## Reproducing

‹TBD — setup, the command that produces each figure, and the run logs.›

## License

MIT. DIAMOND is MIT; Atari ROMs are downloaded with the dependencies and carry their
own terms.
