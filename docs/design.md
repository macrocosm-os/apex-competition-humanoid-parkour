# Design decisions

What the code does is in the code. This records the things that are *not* visible there: what was
measured, what was rejected, and what is still open.

## Robot: G1 12-DoF, and why not 29

The competition ships Unitree's `g1_12dof` — 12 actuated leg joints, with the upper body welded
to the pelvis as rigid mass and collision geometry. The arms are physically present (17.7 kg of
the 32.1 kg total, 29 geoms) and they hit things. They just do not move.

The alternative was the 29-DoF Menagerie G1, which would have allowed arm motion, active vaulting
and true climbing. It was rejected on one measurement: **there is no working policy for it.** The
stock walker transfers to 29-DoF not at all — with the upper body held at neutral across stiffness
kp ∈ {40, 150, 400, 1000, 3000}, every configuration fell within 1.6 s, against 14.5 m upright on
the 12-DoF model. The gap is embodiment (+3 kg, CoM 2.4 cm higher, 17 extra articulated masses),
not a tuning knob.

**This is a one-way door and should be understood as one.** Arms are not "unlockable" later:
`g1_12dof.xml` has no arm joints to enable, and switching model changes `nq`/`nv`/`nu`, which
changes the observation and action dimensions, which invalidates every submitted policy and
requires a new `(id, version)`. Shipping 12-DoF means legs-only until the competition is replaced.

The trade taken: a demonstrated-solvable launch with a real baseline, over an unverifiable launch
with arms.

**Decided: launch without arms.** If the competition stagnates — miners converge and the frontier
stops moving — arms are the escalation lever, shipped as a new competition rather than an update
to this one (the interface break makes that unavoidable). Until then, every obstacle is a leg
maneuver and the docs say so plainly.

The corollary is that obstacle sizing must be audited against *leg* capability, not against a
robot with hands. Measured, for the record:

| | |
|---|---|
| leg kinematic reach | 1.30 m (hip pitch spans ±2.88 rad) |
| knee torque limit | 139 N·m |
| hurdle | 0.62 m — 2.1x reach margin |
| step-up | 0.55 m — needs ~31-63 N·m, so 2.2-4.5x torque margin |
| duck bar | 1.05 m vs 1.26 m standing height — a ~0.2 m squat |

This audit is why the duck bar moved 0.75 m -> 1.05 m, and it is the check that was skipped on the
other two segments in v0.3.0: they shipped named "vault" and "climb-up", words that presuppose
arms. Renamed in v0.3.3; the geometry was fine, the names were not.

## On-ramp calibrated against a real policy, not taste

Difficulty was set by driving the stock walker over candidate geometry:

- it **climbs 15.4° and stalls at 20.1°**, so the on-ramp sits at 15.4° — the steepest short climb
  a naive policy can still manage;
- **drop height is nearly free**: 0.20 m and 0.55 m end its run in the same place, because a
  flat-ground walker has no landing controller at all. The on-ramp takes the full 0.55 m, since
  the spectacle is free;
- it needs **heading hold** to be usable as a probe — it tracks body-frame velocity with no
  heading feedback and drifts 0.26 m sideways per metre travelled.

That last point cost a misdiagnosis worth recording: an early sweep concluded the walker "falls on
a 0.10 m step" and nearly led to replacing the stairs with a ramp. It was lateral drift off the
track. The tell was the control case — flat ground failed at the identical distance.

## Duck-under at 1.05 m, not 0.75 m

The original design put the overhead bar at 0.75 m. A legs-only G1 cannot clear that: standing
head height is 1.26 m, and a deep squat only brings it to ~0.9 m. A segment no embodiment can
pass is a wall, not an obstacle. 1.05 m forces a ~0.2 m squat-walk, which is achievable and still
reads as a duck on playback.

## Randomised conditions on a fixed course

The single most consequential decision, and the one most likely to be questioned in review.
**It was reversed in 0.4.0**; both halves of the argument are recorded here, because the reversal
is the point.

0.3.3 made the instance suite a pure function of `(index, count)`, ignoring the platform's
per-round seed. The reasoning was:

1. The course is static and public, so a per-round seed buys **no secrecy**.
2. It does buy score noise. Measured per-instance stdev is **0.0176**.
3. The takeover margin is 1% of the baseline: **0.002**.
4. The sizing criterion σ_round ≤ margin/4 would need **~1400 instances**. At ~1.14 ms per control
   step that is ~48 minutes, against a 900 s referee timeout. It does not fit.
5. A fixed suite sets σ_round to **zero** instead. Verified in-image: four different `SEED` values
   all return the same score, bit for bit.

Step 1 is where it goes wrong, and it took a while to see. A per-round seed buys no secrecy *about
the course*, which is true and irrelevant. What a fixed suite gives away is the whole evaluation:
geometry, friction, reset noise and step count are all deterministic and all computable offline
from this repo, so the cheapest route to the top of the leaderboard is to solve 24 known open-loop
control problems offline and replay the trajectories. That is not a variant of the intended
solution, it is a different and cheaper one, and it beats a real policy on the metric while
embodying none of the goal.

0.4.0 draws friction **and** wind per instance from the round seed instead
(`env/sim.instance_spec`). Nothing about the geometry changes — the course stays static and
public, which is what keeps the change small and the difficulty stable. What changes is that the
conditions a submission will be scored under are not knowable when it is submitted, so weights
have to encode a controller rather than a trajectory.

**What this costs, stated plainly: σ_round is no longer zero, and the 0.3.3 sizing argument now
applies to us.** Taking the measured per-instance stdev of 0.0176 as a stand-in, independent
draws over 24 instances give σ_round ≈ 0.0176/√24 ≈ **0.0036**, against a takeover margin of
0.002 — so the margin is *inside* the noise and the top slot will random-walk. Adding wind can
only widen it. Two things make this survivable rather than fatal, and neither is a substitute for
measuring it:

- The platform re-scores the incumbent on the **same round input** as its challengers, so pairing
  cancels the course-difficulty main effect. It does not cancel the policy×condition interaction.
- Only friction and wind vary. Holding geometry fixed keeps that interaction far smaller than it
  would be under course-composition randomisation, where a policy meets obstacles it has never
  seen in an order it has never seen.

**The measurement to run before launch** is σ_round for a policy that actually gets deep into the
course, over ≥20 seeds, and then to set the takeover threshold (or `num_instances`) against it.
The 0.0036 figure above is an extrapolation from a baseline whose variance is bimodal — it either
clears the on-ramp or does not — so it is an order-of-magnitude guide, not a result.

### The friction band is narrow, and both ends are pinned

µ ∈ [0.35, 0.50] is a small window, and it is small because measurement closes it from both sides.
`friction_level` is uniform on [0, 1] and maps linearly across the band, so the band *is* the
difficulty distribution — a wide one spends most draws on grippy surfaces and the seed stops
mattering. The old [0.50, 1.25] put the average instance at µ 0.875.

**The floor is bounded by the course.** The on-ramp is 15.38°, so nothing walks up it below
µ = tan(15.38°) = 0.275, and a walking biped needs real margin over that static figure because it
has to push off and brake as well as stand. Measured against the leaderboard leader over 120
instances, *every* band capped below 0.40 returns **zero completions**, with runs piling up on the
ramp at 5–8 m of 51.14 m. 0.35 is the floor; lower is not harder, it is impossible.

**The ceiling is bounded by variance.** It controls how many instances are easy, so lowering it is
the real difficulty lever:

| ceiling | mean µ | leader completions / 120 | seeds with a completion |
|---|---|---|---|
| 0.55 | 0.442 | 8 (6.7%) | 5 of 5 |
| **0.50** | **0.419** | **5 (4.2%)** | **4 of 5** |
| 0.48 | 0.410 | 1 (0.8%) | 1 of 5 |
| 0.45 | 0.396 | 1 (0.8%) | 1 of 5 |

Below 0.50 it falls off a cliff and whole round seeds return no completion at all. That does not
make the competition harder in a useful way — it makes the top slot depend on which seed was
drawn, which is the σ_round problem above made worse. 0.50 is the last ceiling that stays
robustly completable.

## Wind

Wind is MuJoCo's own fluid model, not an applied-force hack: `opt.wind` is subtracted from each
body's linear velocity and quadratic drag follows from `opt.density`, which 0.4.0 sets to
1.204 kg/m³ (air at 20 °C). Two consequences worth knowing:

- Drag now exists at **zero wind** too, so the baseline score moves even on a calm instance.
- The model infers geometry from per-body equivalent-inertia boxes, with no occlusion — every body
  sees free stream. Summed over this robot that is 0.297 m² of frontal area, so it *overestimates*
  the true drag on a G1. Sizing the band against it is therefore conservative in the right
  direction.

The band is 0–14 m/s, uniform, direction uniform over the full circle. That is Beaufort 7 at the
robot, which is stronger weather than it sounds: forecast wind is quoted at 10 m and drops to
50–70% of that near the ground. It works out to 0.179 N per (m/s)², so **35.1 N at the top of the
band — 11.1% of the G1's 315 N weight**, needing a ~6.7 cm shift of the centre of pressure to
stand against. Sustained, not impulsive: the push-recovery literature's 50–300 N figures are
0.05–0.1 s impulses on adult-size humanoids and are not comparable.

**There is a hard ceiling not far above the band, and it is worth writing down so nobody reaches
for "just make it windier".** Drag scales as v² against a fixed weight, so the µ needed merely to
hold station is 0.179 v²/315. At 22 m/s that is 0.275 and the course stops being completable in
*any* direction, even with maximum grip. Hurricane force (32.7 m/s) needs µ = 0.608 and a 36.5 cm
centre-of-pressure shift against a ~9 cm foot: it is outside the physics rather than merely hard,
and no amount of training reaches it. Measured, the last speed that still yields completions is
20 m/s. Also note `wind_max_ms` caps a *uniform draw* — raising it raises the mean instance, it
does not make every instance windy.

Wind is not in the observation, for the same reason friction is not: feeling a disturbance and
adapting is the skill being paid for. It is reported per instance in post-round metadata.

## Recurrence is required by the design, not a nicety

Friction and wind vary per instance and neither is observable. The only way to adapt to a slick
patch or a crosswind is to remember having slipped or been pushed. So the interface carries an
opaque 256-float state vector, zeroed on reset and threaded by the player between `/act` calls.

This also fell out of the baseline: the stock walker **is** an LSTM. A feed-forward-only contract
would have made the reference policy unrepresentable.

## Submission size cap: 15 MB

**Lowered from 25 MB in 0.5.0, and the reasoning is inverted from what it was.** Every earlier
draft of this section concluded "compute does not bind". That was wrong, and wrong for an
instructive reason: it was measured on the wrong machine.

The old table below was produced by a host-side probe and then re-measured on a GitHub amd64
runner. Neither is the evaluation environment. Measured **inside the player sandbox** on the PR
environment, a 5.89M-param / 22.5 MiB graph costs **2.62 ms/step** — against the 0.352 ms the
table claims for the same architecture class, i.e. **7.5x**. Over a full-survival run that is
~252 s of inference, not 34 s.

Cost is cleanly linear in artifact size. Six models from 0.13 to 22.5 MiB, single-threaded ONNX
Runtime, residuals ≤ 0.012 ms:

| size | params | ms/step (host) |
|---|---|---|
| 0.13 MiB — the reference class | 31k | 0.018 |
| 4.24 MiB | 1.1M | 0.070 |
| 7.80 MiB | 2.0M | 0.122 |
| 14.42 MiB | 3.8M | 0.245 |
| 22.48 MiB | 5.9M | 0.380 |

so `inference ≈ 0.0166 ms/MiB` on a fast host, and ~0.095 ms/MiB in-sandbox — a 5.7x gap that a
shared cloud vCPU accounts for.

Size costs **more than its own inference**, which is the part no earlier draft anticipated. Going
0.13 → 22.5 MiB added 4.89 ms per control step, of which only 2.11 ms was inference. The remaining
~2.8 ms is systemic, most plausibly cache and memory pressure: the player and referee are
scheduled on the same node and compete for the same last-level cache.

Against the referee's 900 s timeout, with 24 instances:

| cap | ms/step | 4000 steps | 3000 steps | 2500 steps |
|---|---|---|---|---|
| 25 MB | 12.6 | 1206 s | 904 s | 754 s |
| **15 MB** | **10.4** | 996 s | **747 s** | 622 s |
| 10 MB | 9.3 | 891 s | 668 s | 557 s |

**15 MB paired with `max_steps_per_episode: 3000` is the chosen point**: ~750 s worst case, ~17%
margin. 15 MB is still ~110x the 0.13-0.14 MB of every Unitree reference walker (G1, H1, H1-2) and
fits a ~3.9M-parameter net, so it does not constrain any architecture this task plausibly wants.

Two things worth recording about *how* this was established, because they bound how much to trust
it. The per-step figures come from single evaluations in a shared environment, and run-to-run node
contention (~2-5 ms/step) is **larger** than the difference between a 15 and a 25 MB cap — one run
with a 14.42 MiB model on a contended node measured *slower* per step than a 22.5 MiB model on an
idle one. So the cap is sized against the **worst observed** per-step cost rather than a fitted
curve, and `max_steps` is the more reliable lever because it bounds the worst case regardless of
what a miner submits. Second, a policy that *completes* terminates early, so the cap only ever
binds the mediocre-but-surviving case — which is exactly the case worth bounding.

The cap used to carry a second justification — a fixed suite means spare parameters invite
memorising 24 instances — which 0.4.0 retires. Worth recording that it was never quantitatively
sound: 24 × 3000 steps × 12 actions is **1.7 MB in fp16**, so the payload a memoriser needs fits
inside any cap this task would plausibly set. Randomising the conditions is what makes memorising
worthless; the cap never did that work.

## The scene is compiled once, not per instance

The G1's collision geometry is 27 STL meshes that MuJoCo converts to convex hulls at compile time.
Building a fresh `MjModel` per instance took the referee to **1098 MiB of a 1.5 GiB limit (71%)**,
against the skill's guidance that the baseline should sit under 50%. It did not OOM, but 438 MiB of
headroom on a hard limit is not somewhere to launch from.

Friction is the only thing that varies between instances, and `geom_friction` is a runtime field,
so the scene is now compiled once and friction written per instance. Peak memory drops to
**560 MiB (36%)**.

Two things this surfaced that are worth keeping:

- The change had to be proved, not assumed. Both paths were run over the full suite and the scores
  compared exactly — bit-identical, `0.2005765356172827` either way.
- Getting there exposed a real latent issue. Friction used to reach MuJoCo through the XML, which
  serialised it at `%.4f`; writing the field directly used full float64 precision, and that
  difference alone moved `raw_score` by **0.15%** — inside the 1% takeover margin. Friction values
  are now explicitly quantised to 4 dp in `sample_frictions` so precision is a stated property of
  the design rather than a side effect of a format string.

It did **not** save wall time, contrary to the expectation that drove the change: 31 s vs 29.8 s
for the suite. MuJoCo evidently caches mesh hull construction across compiles within a process.

## Course friction has to out-prioritise the robot's feet

Writing `geom_friction` on the course is necessary but **not sufficient**. MuJoCo mixes contact
parameters from both geoms in a pair, and for friction the mix is the element-wise **maximum**
whenever the two geoms have equal `geom_priority`. `g1_12dof.xml` declares no geom friction at
all, so the robot's feet take MuJoCo's default of 1.0 — above every µ this course draws. At equal
priority `max()` therefore returns the foot's value on every contact, and the course's band never
reaches the solver.

So the course geoms carry `geom_priority = 1`, MuJoCo's documented mechanism for this case: the
higher-priority geom's contact parameters win outright. It is set once in `_shared_model`, since
priority is constant per geom and only `geom_friction` varies per instance.

Two properties worth keeping in mind when changing anything here:

- **Both sides of a contact matter.** Lowering a course µ does nothing on its own if the other
  geom in the pair sits higher at equal priority. Any future surface — a new segment, a moving
  obstacle, a different robot model — needs the same treatment.
- **Assert on contacts, not on scores.** A score cannot distinguish "the band applied" from "the
  band was mixed away but the policy is robust", so a score-level check is not evidence either
  way. `tests/test_friction_reaches_contacts.py` asserts the solved contact µ tracks the course
  geom's µ, and that lowering the band moves it.

## Rejected

- **Checkpoint scoring.** Continuous progress along a linear course already gives a smooth
  gradient; checkpoints add discontinuities and a tuning surface for no benefit.
- **Observation batching.** Considered for evaluation cost. Unnecessary — worst case is ~258 s on
  a native amd64 runner against a 900 s budget, 3.5× headroom.
- **Hands-on-obstacles rules.** Moot on a legs-only robot, and the contact-based gate it needed
  was fragile. The fall gate is now geometric: pelvis clearance above the surface below it, plus
  an uprightness check.
- **Energy budget / fatigue.** Prototyped and dropped. Robots do not tire while the battery
  lasts, and it added a scoring knob without adding difficulty.

## Open

1. **Does the platform inject a fresh `seed` into the round input every round?** 0.4.0's whole
   anti-memorisation property rests on it: a repeated seed repeats the suite exactly. `seed` is
   `required` in `input.schema.json` so a missing one fails loudly instead of silently freezing
   the suite, and the referee prefers the round input over the platform's `SEED` env — but
   neither guards against the same value being sent twice. **Confirm before launch.**
2. **Does the platform re-score the incumbent leader each round, on the same round input?**
   Now load-bearing rather than a correctness footnote: paired scoring is what cancels the
   round-difficulty main effect that randomised conditions introduce. See "Randomised conditions
   on a fixed course".
3. **What is σ_round for a policy that gets deep into the course?** The number the takeover
   threshold should be set against, and not measurable from the baseline. See the same section.
4. **Is the worker fleet homogeneous in CPU generation?** The remaining risk, now narrowed.

   Across *architectures* scores move by amounts comparable to the 1% takeover margin:
   host-vs-image 0.04%, amd64-vs-arm64 0.12%. So `baseline_raw_score` is measured in the referee
   image on a native amd64 runner, which is what the platform runs.

   Across *machines of the same architecture* it appears to be exact. Two separate CI runs on
   two different GitHub amd64 runners both returned `0.20068353334086175` — bit-identical, not
   merely close. That is encouraging but not conclusive: hosted runners are likely the same CPU
   model, so this shows same-generation reproducibility, not cross-generation. If the worker
   fleet spans generations with different FMA or vector-width behaviour, the same policy could
   score differently on different workers, and nothing on our side fixes that.
