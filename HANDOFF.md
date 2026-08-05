# Handoff — `humanoid_parkour` v0.3.3

For platform review. Covers the build checklist, the places this competition deliberately
deviates from the skill's guidance, and what is measured versus assumed.

## Success statement

**A winning solution is a terrain-aware locomotion policy that reads the height scan and adapts
its gait to what is coming — not one that has memorised a fixed sequence of joint targets.**

Alignment checks, to run against top submissions each round:

0. **Is it a leg maneuver?** Every obstacle must be cleared with legs — the robot has no arm
   joints. If a submission appears to clear the hurdle or step-up in a way that requires arm
   contact, that is a physics exploit, not a solution; check the playback.
1. **Does it use perception?** **Now automated and reported every round** — `metadata
   .perception_ablation` carries `abs_delta`, and `abs_delta ≈ 0` means the submission is not
   reading the height scan. This is the single most diagnostic check, because the fixed course makes
   open-loop replay the main degenerate strategy.

   The referee re-runs 4 of the 24 instances showing the policy the terrain from 24 m further along
   the course (wrapped), instead of zeroing obs `[52:97]`. Mismatch rather than zeroing, because
   this check is public: a policy replaying a trajectory could recognise a block of zeros and fall
   over on cue to fake a delta. A real-but-wrong profile has the right distribution, so a policy
   that reads terrain is actively misled and one that ignores it is untouched.

   Verified to discriminate: **0.0000** for the released baseline, which slices obs to indices 0-49
   (`tools/make_baseline.py`) and never sees the scan, against **-0.1287** for a scan-reading policy.
   Sign is not meaningful — that policy scored *better* on mismatched terrain. Magnitude is.

   The 24 m offset is measured, not chosen by taste. A decoy fails either by matching the real
   profile (reports nothing) or by saturating the scan more than the real profile does (then it is
   as obvious as zeros). Over 64 positions along the course: 24 m gives min |Δ| 0.150 m, median
   0.950 m, never degenerate, and 15.5% mean saturation against the real profile's own 11.9%. An
   earlier +12 m failed both ways — identical flat 0.8 m ground across the whole 0-6 m start
   region, where every current policy actually is, and it ran off the far end past ~39 m into the
   distant floor. `env/sim._obs` wraps the decoy inside the course to prevent the latter.

   Non-ranking: it does not touch `raw_scores`, and the ablation steps are not counted in `steps`.
   Tune with `ablation_instances` (0 disables) and `ablation_offset_m` in the round input.
2. **Does it generalise off-suite?** Score it on friction levels between the 24 evaluated ones, and
   on a mirrored or re-ordered course. Real locomotion transfers; a memorised one does not.
3. **Does it look like locomotion?** Watch the playback (`tools/preview.py --run`). Gait should be
   recognisable and recoverable. Ballistic dives that bank progress before falling clear the metric
   without embodying the goal.

## Deviations from the skill's guidance

### 1. The evaluation suite does not rotate per round — this is the big one

`reference/evaluation-design.md` Defense 1 says to rotate the master seed every round so solutions
cannot overfit a frozen instance set. **This competition does not.** Instances are a pure function
of `(index, count)` (`env/sim.instance_spec`); the platform's `SEED` is ignored for instance
generation.

Why: per-instance stdev is **0.0176** against a 1% takeover margin of ~0.002. By the sizing
procedure that needs **~1400 instances** for σ_round ≤ margin/4 — roughly 48 minutes of referee
time against a 900 s timeout and a 20-minute hard ceiling. It does not fit. A fixed,
friction-stratified suite sets σ_round to **zero** instead: four different `SEED` values reproduce
the score bit for bit.

What it costs, stated plainly: **the instance set is overfittable, and the generator is public.**
`env/course.py` and `env/sim.py` ship in this repo, so a miner can compute the exact friction of
all 24 instances offline. Two consequences worth being explicit about:

- Friction is **not** hidden information in the game-theoretic sense. It is absent from the
  observation vector, so a policy cannot read it at runtime — but it can be baked into weights.
  The "adapt by feel" framing is therefore an *option* the design supports, not a constraint it
  enforces.
- The defence against memorisation is not secrecy, it is the alignment checks above plus the
  25 MB cap. That is weaker than seed rotation and it should be reviewed as such.

The variance problem also eases as the field improves: the baseline's variance is bimodal
(it either clears the on-ramp or does not), whereas a policy that reliably completes varies only
in speed. If measured σ_task for a completing policy is low enough, per-round friction rotation
becomes affordable at N=24 and should be reconsidered then. **This is the recommended follow-up.**

### 2. `submission_reveal_days: 5` (production default is 1, range 1–7)

Trained locomotion policies carry real R&D. Per the skill's own guidance ("4–7 days where a
winning solution embodies real IP"), 5 days sits in range.

### 3. No stage validation yet

The full loop has been exercised locally by hand and in CI, but not on stage. That is the one
checklist item this repo cannot close on its own.

## Measured

All numbers from the referee image, at `fixtures/input.json`
(`num_instances: 24`, `max_steps_per_episode: 4000`, `deadline_ms: 500`).

| | |
|---|---|
| `baseline_raw_score` | **0.2007** (native amd64; exact `0.20068353334086175`) |
| completions | 0 of 24 — furthest 10.73 m of 51.1 m |
| eval wall time | 66 s amd64 (31 s arm64); ~258 s worst case vs 900 s timeout |
| referee peak memory | **560 MiB of 1536 (36%)**, measured under `--memory 1.5g` |
| player peak memory | 30 MiB of 1536 (2%) |
| per-instance stdev | 0.0176 |
| determinism | bit-identical across seeds, and across two separate amd64 CI runners |

Architecture sensitivity, all inside the 1% takeover margin — which is why the spec figure is
pinned to amd64-in-image:

| measured | raw_score |
|---|---|
| referee image, native amd64 — **the spec figure** | 0.20068 |
| host, no container (arm64) | 0.20058 |
| referee image, arm64 build | 0.20044 |

## Security checklist

Walked against `reference/security-checklist.md`:

- **§1 revealed data** — post-round metadata is per-instance score, friction level, terminal reason,
  progress, distance, steps. `friction_level` is derivable from public code (see deviation 1), so
  revealing it leaks nothing that is not already computable; it is kept because it is useful for
  debugging. No scoring internals beyond what the published spec and repo already contain.
- **§2 cross-miner** — solo competition. Referee builds fresh `MjData` per instance and holds no
  state keyed on anything a submission controls. The player zeroes policy state on every `/reset`,
  so memory cannot carry across instances (which would otherwise let a policy count episodes and
  infer its position in the friction sweep).
- **§3 data into the player** — the referee passes `seed=0` and `config={}` into `player.reset`
  deliberately: nothing identifying the instance crosses the boundary. The observation carries no
  friction, no segment identity, and no obstacle oracle.
- **§4 internet** — `allow_internet: false`. `network_disabled: false` only so the referee can
  reach the player on the per-job network.
- **§5 persistence** — nothing written outside `/data`; no caches, no warm-up state.
- **§6/§7 screening** — `artifact_type: onnx`, Layer-1 structural validation only. No Layer-2
  image, which is the outcome the skill steers toward: an ONNX graph cannot carry arbitrary code,
  and interface violations are a typed rejection in the player's loader.
- **§8 Goodhart** — gates zero out `physics_glitch` (NaN/Inf state, |qvel| > 100) and
  `out_of_bounds` (|y| > 1.2, so no walking around the course). `invalid_action` covers NaN and
  malformed actions; both paths are tested. Residual hole worth knowing: `progress` uses
  `max_x`, so a ballistic dive banks distance before falling. Bounded (a fall terminates, so
  it buys one dive) and it is what the alignment checks are for.
- **§9 determinism** — exact dependency pins, single-threaded ONNX Runtime, fixed suite. Verified
  bit-identical within an image; see the architecture table for cross-arch behaviour.

## Known follow-ups

1. **Reconsider per-round friction rotation** once a completing policy exists and its σ_task can
   be measured (see deviation 1). This is the highest-value change to the design.
2. **Stage validation.**
3. **Fleet CPU homogeneity** — the open platform question. Same-generation reproducibility is
   demonstrated; cross-generation is not, and it cannot be tested from this repo.
