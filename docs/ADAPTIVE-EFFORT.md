# Adaptive effort on a non-uniform stream

Programme spec, written 2026-09-24. Spans two repos and two seats. Canonical copy lives
here; [RLCD](https://github.com/maskedband1t/RLCD) points at it.

## The claim

> A generative model emits a sequence. A per-step decision allocates effort to each step.
> Effort saved is the payoff; fidelity lost is the risk. **The field ships a hand-tuned
> constant.** This programme measures what a calibrated number is worth in that seat,
> against the constant and against an oracle.

Two demonstrations, one computation.

| | **Seat 1** — skip gate | **Seat 2** — adaptive denoising |
|---|---|---|
| generator emits | a chunk of predicted actions | a frame |
| per-step decision | execute fully, or skip ahead | full denoising steps, or fewer |
| effort saved | robot seconds, operator seconds | denoising steps, fps |
| risk of skimping | an unsafe action | a corrupted frame, trained on as if true |
| risk of never skimping | speed left on the table | compute burned on trivial frames |
| the shipped constant | **2 cm / 0.3 cm** displacement, hand-tuned per scene | **3** denoising steps, uniform per frame |
| substrate | sorting cell, humanoid fetch room | DIAMOND Atari, then MIRA / Rocket Science |
| prior art it answers | [Argon Robotics](https://www.argonrobotics.ai/research/scaling-and-speeding-up-robots-in-the-real-world), Lever 2 | DIAMOND's shipped sampler config |

Argon supplies the method, not a citation. Their finding — uniform frame-dropping takes
plate picking from 94 % to 53 % at 2x and 9 % at 4x, so you must segment the stream before
you accelerate it — is the shape of both seats. Their segmenter is a hand-written
gripper-event rule; ours is a calibrated number. Their optimum sits at 4x for reasons they
state plainly that they do not know.

## The shared skeleton

Identical in both seats. Only the nouns change.

**Arms.**

| arm | what it is | role |
|---|---|---|
| **U — uniform** | the same effort on every step | what ships today |
| **R — rule** | a hand-tuned threshold on an observable proxy | the dumb baseline, and a strong one |
| **C — calibrated probe** | threshold on `P(safe)` / `P(easy)`, from a head on **frozen** features | is the signal there at all? |
| **P — post-trained** | the same probability, from a head **trained into the generator** on its own failures | does teaching the model to judge itself buy anything? |
| **O — oracle** | knows the true difficulty | the ceiling |

**The comparison that matters is C → P.** C asks whether a usable signal exists anywhere in
the frozen system. P asks whether post-training the generator to assess itself beats reading
it from outside. Both answers are worth having: if P >> C, post-training earns its cost; if
P ~ C, a cheap external probe suffices and that is the deployable result.

R is not a strawman in either seat. Seat 1's R is Argon's deployed rule. Seat 2's R is a
step counter, which [amendment 6](../PREDICTIONS.md) found beats every internal signal the
model carries — and a fixed horizon *is* a step counter.

**Metric, reported both ways.** Fidelity at fixed effort, and effort at fixed fidelity.
The second is Argon's own framing (speed at held success), so seat 1 reports in their units
and is directly comparable to their table.

**Matched budget.** Every arm is compared at equal *mean* total effort. An arm that wins by
spending more has not won.

**The permutation control.** Shuffle outcome labels within each value of the proxy R uses.
That destroys content signal while leaving the proxy intact. Under the shuffle, R must be
unchanged and C must collapse to R. If C keeps an advantage, the fit leaked the proxy and
**the result is reported as void**, not refitted.

**Noise floor first.** No small difference is claimed before the repeat-run spread of each
metric is measured and published.

## Why post-training is affordable here: the labels are free

Arm P needs supervision, and in both seats the system produces it by failing.

| seat | one label is | produced by |
|---|---|---|
| 1 | a skip that turned out unsafe | running the bench |
| 2 | a frame that left the noise-floor band | running the harness |

No annotation, no collected dataset, no human in the loop. **The generator labels itself by
being wrong**, and the head is trained on that record. This is the sibling repo's "every
takeover is a free label," restated for effort allocation rather than for handoff — and it
is the only reason arm P fits inside a three-day budget at all.

The arc is identical in both seats, and it is the arc of RLCD:

> the generator cannot judge itself -> collect its own failures -> post-train a head on that
> record -> gate on the head's calibrated output.

## Seat 1 — the skip gate

*Benches already exist; this adds one arm to infrastructure built for RLCD.*

- **Stream:** action chunks on the sorting cell and the humanoid fetch room, both with a
  person in the scene.
- **Effort:** wall-clock task time and operator seconds.
- **Risk event:** unsafe action — contact with the person, a wrong hand-over, a dropped item.
- **R:** Argon's two-threshold displacement rule, transposed to the bench's units: one
  threshold for the clean scene, a tighter one when a person is present.
- **C:** one calibrated `P(safe to skip)` read from frozen features, **one threshold for
  both scenes.**
- **P:** the same probability from a head post-trained on the bench's own unsafe skips —
  the 421M-owned-copy recipe from RLCD, pointed at skip safety instead of handoff.

**Headline question.** Argon needed two numbers because one did not transfer between scene
types. Does a single calibrated threshold match both hand-tuned ones, with no per-scene
switch? That is the transfer claim, and it is the whole argument for a calibrated number
over a tuned constant.

## Seat 2 — adaptive denoising

- **Stream:** generated frames from DIAMOND's diffusion world model. Breakout and Pong.
- **Effort:** denoising steps per frame.
- **Risk event:** the frame leaves the noise-floor band — divergence, as already defined.
- **U:** 3 steps on every frame, DIAMOND's shipped default.
- **R:** the step counter.
- **C:** a self-assessment head **post-trained into the model**, predicting per-frame
  difficulty from the divergence labels the existing harness already produces.

**Feasibility gate, runs first, ~20 minutes of GPU.** Sweep uniform denoising steps at
1 / 2 / 3 / 5 / 10 and measure fidelity. **If that curve is flat, there is nothing to
reallocate and seat 2 is dead** — report it and stop. This is deliberately the cheapest
possible way to kill the seat, and it runs before anything is built.

**Arm P here is a head fine-tuned into the denoiser**, trained on divergence labels the
existing harness already emits. Amendment 6 found the model carries no internal signal about
its own error. The response is not "so nothing works" — it is *teach it*.

**Cost asymmetry, stated plainly.** Seat 1's arm P is cheaper: the distillation recipe
already exists in RLCD, the benches are simulated, and labels are generated in minutes.
Seat 2's arm P means fine-tuning a diffusion denoiser, which is heavier. So **seat 1 is the
faster route to a post-training result, and seat 2 is the one that says "world model" on the
tin.** That trade, not the science, is what decides which seat runs first.

## What would falsify the unification itself

Recorded before either seat has run, because a programme claim needs a failure condition
as much as an experiment does:

> **If the calibrated gate wins in one seat and loses in the other, "same computation" is
> too strong.** The claim is then narrowed, in public, to whichever seat it holds in, and
> the difference between the seats becomes the finding.

Candidate reasons that could happen, named now: seat 1's risk events are rare and
catastrophic while seat 2's are common and graded; seat 1's proxy is a physical quantity
while seat 2's is a clock; seat 1 has a human in the loop to absorb a bad call and seat 2
does not.

## Status

| | seat 1 | seat 2 |
|---|---|---|
| benches / harness | exist (RLCD) | built, CPU-verified, unrun |
| pre-registration | to write | P1–P10 registered |
| pilot | none | amendments 6 and 7 |
| arm P cost | low — recipe exists, sim labels | higher — denoiser fine-tune |
| blocking step | none — bench time | the feasibility gate above |

One seat is taken to completion first. The other stays specced and public, so the line
reads as one programme with an arm outstanding rather than two projects that drifted.
