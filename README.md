<img width="1267" height="625" alt="01_overview_iso" src="https://github.com/user-attachments/assets/17b9f4ca-39ef-4d15-a216-b3eee1888a43" />

# Humanoid Parkour

An Apex competition (Bittensor Subnet 1). Miners submit an **ONNX policy** that drives a Unitree
G1 humanoid through a 51 m parkour course: a steep on-ramp, a sheer drop, stairs, a 1 m leap over
a real void, a hip-high hurdle, a 0.55 m step-up, a duck-under, a balance beam, a slick patch,
and a stairway down.

**Nobody has finished it.** The reference policy — Unitree's own stock G1 walker — gets 21% of
the way and falls off the first ledge.

| | |
|---|---|
| id / version | `humanoid_parkour` 0.5.0 |
| robot | Unitree G1, **12 actuated leg DoF only** — no arm joints, 32.1 kg |
| submission | ONNX graph, ≤ 15 MB, architecture free |
| interface | `obs[104]` + `state_in[256]` → `action[12]` + `state_out[256]`, float32 |
| evaluation | 24 instances, ≤ 3000 control steps each (60 s sim), conditions drawn per round |
| baseline | see [`baseline/PROVENANCE.md`](baseline/PROVENANCE.md) |

## The robot has no arms

All 12 actuators are legs (hip pitch/roll/yaw, knee, ankle pitch/roll, ×2). The arms are 17.7 kg
of collision geometry welded to the pelvis: they are present, they have mass, and they hit things
— but nothing can move them. **Every obstacle is a leg maneuver.** There is no vaulting, no
pulling up, and no arm swing for balance.

Both tall obstacles are sized against measured leg capability, not against a robot with hands:
the leg reaches **1.30 m** kinematically (hip pitch spans ±2.88 rad) against a 0.62 m hurdle, and
the 0.55 m step-up needs ~31-63 N.m at the knee against a **139 N.m** limit.

## The course

51.1 m, linear, on a raised plinth so gaps are real voids. Difficulty ramps along its length, so
progress-based scoring gives a continuous gradient rather than discrete tiers — a policy that
gets 3 m further scores 3 m better, all the way along.

| Maneuver | Geometry |
|---|---|
| on-ramp | 6 m flat, 15.4° climb over 2 m, then a 0.55 m sheer drop |
| stairs up / down | rise 0.18–0.20 m, run 0.32–0.34 m |
| leap | 1.0 m void |
| drop-down | 0.6 m |
| hurdle | 0.62 m barrier — stepped over, not vaulted |
| step-up | 0.55 m platform (70% of hip height) |
| duck-under | overhead bar at 1.05 m — forces a ~0.2 m squat-walk |
| balance beam | 0.32 m wide, 3.5 m long |
| slick patch | low friction, geometry identical to flat |

The geometry is **static and public**. What varies between instances is **surface friction and
wind**, both drawn at random from a per-round seed, and neither is **observable** — a policy has
to feel the slip or the push and adapt rather than read a number. That is why the interface
carries recurrent state.

| Condition | Range, per instance |
|---|---|
| friction | µ ∈ [0.35, 0.50] course-wide, ±8% per-slab jitter (wet tile → wet concrete) |
| slick patch | µ ∈ [0.08, 0.14] (near-ice → wet tile) |
| wind | 0–14 m/s, any direction in the horizontal plane; steady for the episode |

14 m/s is Beaufort 7 *at the robot*, worth 35.1 N of drag — 11.1% of the G1's weight, pushing
sideways for the whole run.

The friction band is deliberately narrow and low. `friction_level` is uniform on [0, 1] and maps
linearly across it, so a wide band spends most of its draws on grippy surfaces; narrowing it is
what makes every instance a friction problem. Both ends are pinned by measurement: the on-ramp is
15.38°, so nothing walks up it below µ = tan(15.38°) = 0.275 and every band capped under 0.40
yields zero completions in 120 instances — while a ceiling under 0.50 drops the completion rate
below 1%, which measures round variance rather than skill.

```bash
python -m env.course          # print the layout
python tools/preview.py --seed 1   # stills + flythrough (needs mujoco + ffmpeg)
```

## Scoring

Per instance, higher is better:

| Outcome | Score |
|---|---|
| completed | `1.0 + (max_steps - steps) / max_steps` → (1.0, 2.0] |
| fell / timeout / out_of_bounds | `progress`, the fraction of the course covered → [0.0, 1.0) |
| physics_glitch / invalid / player error | 0.0 |

`raw_score` is the mean over the 24 instances. Any completion outranks any non-completion, faster
completions outrank slower ones, and partial progress gives non-finishers a training gradient.

## Why the conditions are randomised

The course you train against is the course you are scored on — that part is fixed and public on
purpose. What you **cannot** know in advance is the friction and wind of the 24 instances: they
are drawn from a per-round seed (`env/sim.instance_spec`) that is not published while the round
is open.

This matters because of what the alternative was. Up to 0.3.3 the suite was a pure function of
`(index, count)`, which made the entire evaluation — geometry, friction, reset noise, step counts
— reproducible offline, bit for bit, from this repo. The cheapest way to the top of the
leaderboard was then not to learn to walk but to optimise 24 open-loop joint trajectories offline
and replay them. Randomising the conditions is what makes that worthless: a replayed trajectory
meets a surface and a crosswind it was not optimised for and falls over.

The cost is honest and worth knowing if you are chasing the top slot: scores now carry
round-to-round noise, so a marginal improvement may not resolve in a single round. Conditions are
reported per instance in post-round metadata, so you can see exactly what you were scored on.

**Practically:** train across the full ranges above, not at their midpoints. A policy tuned for
µ = 0.9 in still air will meet µ = 0.55 with a 7 m/s crosswind.

## Perception

The policy sees proprioception (projected gravity, base velocities, joint angles and velocities,
its own last action, a gait clock), its pose on the track (heading, lateral offset, distance to
the finish), and terrain:

- **height scan** — a 9×5 grid of downward samples in the robot's yaw frame, from 0.4 m behind to
  1.6 m ahead, given relative to the pelvis
- **overhead clearance** — 7 upward samples ahead, which is how the duck-under is visible

It does not get an obstacle oracle. There is no "a leap starts in 1.2 m" channel, no segment
identity, no friction and no wind. Full layout in [`env/sim.py`](env/sim.py).

## Watching an evaluation

A score says how a run ended, not how it got there. Every evaluation therefore emits two
artifacts alongside `result.json`, both collected by the platform and listed on the submission:

| artifact | what it is | where it comes from |
|---|---|---|
| **history** — 24 × `instance_NN.json` | the robot's pose and the policy's action at every recorded step, plus the friction and wind the instance was drawn with | the referee writes `/data/history/`; collected as `FileType.HISTORY` |
| **log** — one line per API call | every `/health`, `/reset` and `/act` the referee made, with latency and status | the player sandbox's stdout; collected as `FileType.LOG` |

Miners download both after the round (`eval_file_paths` on the submission), and
`tools/replay.py` plays a history file back:

```bash
PYTHONPATH=. python tools/replay.py downloaded/            # list the instances
PYTHONPATH=. python tools/replay.py downloaded/ --worst    # film the run that scored lowest
PYTHONPATH=. mjpython tools/replay.py downloaded/ -i 7 --live   # interactive viewer
```

Locally, `--record DIR` writes the identical format, so the same tool reads both:

```bash
PYTHONPATH=. python tools/local_eval.py baseline/baseline.onnx -n 24 --record runs/base
```

Replay is **MuJoCo only** — no policy, no onnxruntime, no physics. It sets `data.qpos` and calls
`mj_forward`, which recomputes everything a renderer needs, so a run stays viewable after the
submission and the round seed are gone, and cannot drift from what was scored the way
re-simulating from the action log could. The actions are stored as diagnostics, not for replay.

Recording costs one array copy per step and does not touch scoring — a recorded suite produces
byte-identical numbers to an unrecorded one, and a history write that fails is logged and ignored
rather than failing the round. Set `record_history: false` in the round config to turn it off.

Sizing, at the default stride 2 (25 Hz, which is what the mp4 renders at anyway, so the video is
identical to recording at 50 Hz):

| | per instance | per 24-instance round |
|---|---|---|
| history, run to the 3000-step cap | ~264 KiB | ~6.3 MB |
| history, typical baseline run (falls ~20 s) | ~86 KiB | ~2.1 MB |
| API log | — | ~2.7 MB typical, ~10.8 MB worst case |

`--record-stride N` (or `history_stride` in the round config) trades resolution for size; 1 keeps
all 50 Hz for slow-motion. `APEX_API_LOG=0` disables the API log.

`--live` needs `mjpython` rather than `python` on macOS, because the passive viewer has to own the
main thread there. Filming to mp4 is headless and needs only ffmpeg.

## Submitting

The tensor signature is fixed; the graph is not — recurrent nets, ensembles, transformers over a
history window. The 15 MB cap is ~110x the size of the reference policies it is measured against
(every Unitree humanoid walker is 0.13-0.14 MB), so it is unlikely to be what stops you.
`state_in`/`state_out` are your own opaque per-episode memory — zeroed on reset, fed back each
step. A feed-forward policy ignores `state_in` and returns zeros.

**The cap is a wall-clock limit, not a taste judgement.** Inference runs once per control step, up
to 72,000 times per evaluation, and its cost is linear in artifact size — measured on evaluation
hardware, a 22.5 MiB graph costs 2.62 ms/step against 0.51 ms for the 134 KB reference walker.
Lowered from 25 MB in 0.5.0 because the referee has a 900 s timeout and a max-size policy that
survived every step overran it.

Off-the-shelf G1 locomotion policies are a reasonable starting point and that is exactly what the
baseline is, but none of them can see terrain, so none of them will get past the on-ramp without
retraining.

## Resource use

Measured under the spec's own limits (`--cpus 1 --memory 1.5g`): referee peaks at **560 MiB of
1536 (36%)**, player at 30 MiB. Referee memory does not grow with submission quality, so that is
near the true ceiling rather than a floor better solutions will grow into.

## Repo layout

```
env/            course, physics, perception, gates, scoring, history format (referee image)
  assets/       vendored Unitree G1 12-DoF model + collision meshes (BSD-3)
player/         ONNX serving + interface validation           (player image)
referee/        match driver, fault attribution
baseline/       the reference policy and where its number came from
tools/          baseline export, local eval, course preview, history replay
docs/           design notes
spec.yaml       the competition manifest
```

## Running it end to end

```bash
docker build -f referee/Dockerfile -t hp-referee .
docker build -f player/Dockerfile  -t hp-player  .

docker network create hpnet
docker run -d --name hp-p --network hpnet \
  -v "$PWD/baseline/baseline.onnx:/app/submission.onnx:ro" hp-player

mkdir -p /tmp/hpdata && chmod 777 /tmp/hpdata
docker run --rm --network hpnet -v /tmp/hpdata:/data \
  -e MATCH_ID=local -e SEED=1 -e NUM_PLAYERS=1 -e PLAYER_URLS=http://hp-p:8000 \
  -e CONFIG_JSON='{"seed":1,"num_instances":24,"max_steps_per_episode":3000,"deadline_ms":500}' \
  hp-referee
jq '.raw_scores, .metadata.num_completed' /tmp/hpdata/result.json
```

Takes ~67 s with the baseline on a worker-class amd64 CPU, and ~258 s worst case for a policy that
survives every step, against the referee's 900 s timeout. (An arm64 build of the same images runs
it in ~30 s, and scores 0.12% differently — which is why `baseline_raw_score` is measured on
amd64; see [`baseline/PROVENANCE.md`](baseline/PROVENANCE.md).)

## History

An earlier, much simpler version (flat plane, step-over hurdles, Gymnasium humanoid) was built,
released and signed as **`v0.2.0`**. It is preserved at the
[`v0.2.0`](https://github.com/macrocosm-os/apex-competition-humanoid-parkour-v2/tree/v0.2.0) tag
with its images in GHCR, and is not part of this codebase. The predecessor repo
`apex-competition-humanoid-parkour` is archived.
