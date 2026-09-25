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

## Three constraints from the sibling programme

Added 2026-09-25 from results in the human–robot seat. Each one would have invalidated a
headline number here if it had been discovered after the runs instead of before.

### 1. Arm P must be scored on a bank its corrections never touched

Post-training a head on correction data makes its probability honest **where you corrected
and dishonest where you did not**. Measured in the sibling repo: calibration error went
.362 → .008 on corrected lines and .307 → **.399** on an untouched bank over the same six
rounds, while the operator's veto window fell from rescuing 34 of 60 lines to rescuing 0.

This attacks seat 1 at exactly its claim. If arm P is post-trained on one scene type's
unsafe skips, it will look excellent there and fail quietly on a scene type it never saw —
which is the *transfer* the experiment exists to demonstrate.

> **Required in both seats:** every round scores a held-out bank the corrections never
> touched, and that bank's calibration is reported beside the corrected one. A falling
> failure rate on covered states is not evidence the gate transfers.

### 2. The label form is a design variable, not a property of the model

Same correction records, three ways of writing down what the operator did, scored on 60
lines the corrections never touched: a head trained on the operator's **replacement action**
rescued **+0**; trained to **ask**, +3; trained on **vetoes**, **+14**, and with the best
calibration of the three (.261 against .376), at about 24 operator seconds per line.

So arm P's survivability off-distribution is something the experiment *chooses*. Seat 1
trains on vetoes — a veto is the natural label for "that skip was not safe" anyway. Seat 2's
analogue is to label on *the frame that broke*, not on the corrected continuation.

### 3. Every speed comparison must be paired

A 2.3x speed-up in the sibling programme evaporated under pairing: failures are the
expensive episodes, **122 s against 25 s**, so an unpaired mean measures the failure rate
rather than the decision speed. Logged there as method error 53.

The arms here will have different success rates by construction, so the same trap is live.

> **Required:** speed and effort comparisons are paired on episodes *both* arms completed
> successfully. The unpaired average is reported separately and labelled a fleet cost, never
> as the speed result.

This tightens the "matched budget" rule stated above, which was necessary but not sufficient.

## A prior that is not a finding

Seat 1's bench assumes a teleop-to-autonomy fleet where one operator supervises several
robots and the handoff decision is the commercial product. **That assumption comes from
public job postings and the shape of the field, not from any verified source.** It is a
prior. Nothing in this spec should be read as a claim about how any particular company
operates, and the bench must not inherit it as fact.

### 4. Which pairing regime is the bench in? Check, do not assume

Added 2026-09-25. An audit of 88 arm pairs in the sibling programme found the failure-cost
asymmetry is **a property of embodied episodes, not of the decision layer**. On a
decision-level bench with no physics, a miss cost 10.2 s against 11.1 s for a success —
failing costs no time, so pairing barely moves the numbers. On embodied benches the same
comparison ran 43 s unpaired against 24 s paired.

So constraint 3 bites hard or barely, depending on the bench, and seat 1's is decision-level
in parts and embodied in others.

> **Run first, before any arm comparison:** within each arm, compare episode time on handled
> against missed episodes. That single check says which regime the bench is in. Two lines of
> analysis, and it decides whether an unpaired number is merely imprecise or actively
> backwards.

The audit also found genuine sign flips in the embodied set — one comparison read 10 s slower
unpaired and 2 s faster paired. Direction, not just magnitude, can invert.
