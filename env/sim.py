"""MuJoCo simulation of one Humanoid Parkour episode.

Robot: Unitree G1, 12 actuated leg DoF (`env/assets/g1_12dof.xml`, vendored from unitree_rl_gym,
BSD-3). The upper body is real mass and real collision geometry welded to the pelvis — the arms
are there and they hit things, they just are not actuated. Torque actuators driven by a PD loop
in `step`, so the ACTION IS A JOINT POSITION TARGET, not a torque.

Deterministic by construction: same (instance seed, action sequence) -> same trajectory.
The referee owns the physics; the player only ever sees the observation vector.

Observation (float32, OBS_DIM = 104), all in the robot's yaw frame unless noted:
    [0:3]     projected gravity                        (which way is down, from the body)
    [3:6]     base angular velocity      * 0.25
    [6:9]     base linear velocity       * 2.0
    [9:21]    joint angles - default pose
    [21:33]   joint velocities           * 0.05
    [33:45]   previous action
    [45:47]   gait clock, sin/cos of a 0.8 s cycle
    [47:49]   heading error, sin/cos of yaw (course runs along +x)
    [49]      lateral offset y
    [50]      distance to the finish line / 10
    [51]      pelvis height above the surface directly below
    [52:97]   height scan, 9 x 5 grid, surface height relative to the pelvis
    [97:104]  overhead clearance, 7 samples ahead

Action (float32, ACT_DIM = 12): joint position targets as offsets from the default pose,
scaled by ACTION_SCALE and clipped to the joint limits.

Surface friction varies between instances and is NOT in the observation. Adapting to a surface
you cannot see is the point (env/course.py). The COURSE is drawn from the platform's per-round
seed, so its layout is not knowable before the round opens; the instance suite within a round is a
fixed, stratified friction sweep over that one course — see `instance_spec`.

Termination gates (each maps to a terminal_reason the miner sees post-round):
    completed       pelvis past the finish line
    fell            pelvis under FALL_CLEARANCE above the surface below it, or torso past ~66 deg
    out_of_bounds   |y| > TRACK_HALF_W (no walking around the course)
    physics_glitch  NaN/Inf state or |qvel| > 100 (glitch-surfing scores 0)
    timeout         max_steps control steps elapsed
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

from .course import (GEOM_PREFIX, PLINTH_TOP, TRACK_HALF_W, WORLD_GROUP, Course,
                     course_xml_fragment, generate_course, sample_frictions)

ASSETS = pathlib.Path(__file__).parent / "assets"

ACT_DIM = 12

# Opaque per-episode policy memory, threaded by the player between /act calls and zeroed on
# /reset. Recurrence is not a luxury here: surface friction is randomised and NOT observable, so
# a policy can only adapt to a slick patch by remembering that it just slipped. A feed-forward
# submission simply returns zeros and ignores it. The stock-walker baseline is itself an LSTM.
STATE_DIM = 256

SCAN_NX, SCAN_NY = 9, 5          # height-scan grid, 45 rays
OVERHEAD_N = 7
OBS_DIM = 52 + SCAN_NX * SCAN_NY + OVERHEAD_N   # = 104

# Where the scan looks, in metres in the robot's yaw frame. Backwards a little so the policy can
# see the edge it is standing on, forwards far enough to plan a leap.
SCAN_X = np.linspace(-0.4, 1.6, SCAN_NX)
SCAN_Y = np.linspace(-0.5, 0.5, SCAN_NY)
OVERHEAD_X = np.linspace(0.0, 1.8, OVERHEAD_N)
SCAN_CLIP = 1.0                  # scan values saturate here, in metres

PHYS_DT = 0.002
FRAME_SKIP = 10                  # 10 x 0.002 s = 20 ms per control step (50 Hz)
DEFAULT_MAX_STEPS = 4000         # 80 s of sim time
ACTION_SCALE = 0.25
QVEL_GLITCH_LIMIT = 100.0
RESET_NOISE = 0.01
FALL_CLEARANCE = 0.45            # pelvis this far above the surface below, or it has fallen
UPRIGHT_MIN = 0.40               # projected-gravity z; ~66 deg of tilt
RAY_FROM_ABOVE = 3.0             # height above the pelvis to cast downward scan rays from

# Unitree's PD gains and home pose for this robot (deploy/deploy_mujoco/configs/g1.yaml).
KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float64)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float64)

START_X = -0.8
GAIT_PERIOD = 0.8


class InvalidAction(ValueError):
    """The action was not a finite ACT_DIM vector."""


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    terminal_reason: str | None  # None while the episode is still running


def _scene_xml(course: Course, frictions: list[float] | None) -> str:
    """The robot model with the course spliced into its worldbody."""
    robot = (ASSETS / "g1_12dof.xml").read_text()
    floor = (f'    <geom name="floor" type="plane" size="80 20 0.1" pos="30 0 0" '
             f'condim="3" group="{WORLD_GROUP}" rgba=".18 .19 .22 1"/>\n')
    start = (f'    <geom type="box" pos="{START_X - 0.7:.3f} 0 {PLINTH_TOP - 0.2:.3f}" '
             f'size="1.5 {TRACK_HALF_W} 0.2" condim="3" group="{WORLD_GROUP}" '
             f'friction="1 .1 .1" rgba=".45 .47 .5 1"/>\n')
    body = floor + start + course_xml_fragment(course.segs, frictions)
    return robot.replace("</worldbody>", body + "\n  </worldbody>")


def instance_spec(i: int, n: int) -> tuple[float, int]:
    """The (friction level, instance seed) of evaluation instance `i` of `n`.

    A pure function of (i, n): within a round, the COURSE carries the per-round randomness (it is
    drawn from the platform's master seed in `env.course.generate_course`) and the instance suite
    sweeps friction across that course. Keeping the sweep deterministic means every submission in
    the round runs exactly the same instances, so identical resubmissions score identically.

    Coverage comes from stratification instead of randomness. Levels are spread evenly across
    the friction range, so N instances sample the whole grippy->slippery continuum rather than
    clustering wherever a draw happened to land.

    Note what this does NOT fix. With one course per round, round-to-round score variance is a
    single course draw and is therefore large: the v0.2.0 design measured sigma_round = 0.0304
    against a 1% takeover margin (17x too noisy, `variance_baseline_N120_image.json`). Averaging
    several courses per round is the lever that shrinks it, at the cost of one model compile each.
    """
    return (i + 0.5) / n, i


_MODEL: mujoco.MjModel | None = None
_MODEL_KEY: tuple | None = None
_COURSE_GEOMS: list[int] = []


def _course_model(course: Course) -> tuple[mujoco.MjModel, list[int]]:
    """Compile the scene once per course and reuse it for every instance on that course.

    Compiling costs ~1.1 s and a few hundred MB, because the G1's collision geometry is 27 STL
    meshes that MuJoCo converts to convex hulls. Doing it per instance was ~26 s of a 67 s
    evaluation and pushed the referee to 1.1 GB against a 1.5 GiB limit — so the cache is what keeps
    per-round generation affordable: one course per round means one compile per round.

    Nothing instance-specific is baked into the model: friction is the only thing that varies within
    a round, and `geom_friction` is a runtime field, so it is written per instance in __init__.
    Instances run strictly sequentially and each gets a fresh MjData, so sharing the model is safe.

    If this ever needs to hold SEVERAL courses per round, do NOT mutate geometry on a model compiled
    for a different course: `geom_pos`/`geom_size` are runtime fields, but MuJoCo derives the
    broadphase BVH (`bvh_aabb`, `bvh_nodeid`) at compile time, and a stale BVH silently DROPS
    contacts while the height scan still reads the new geometry perfectly. Measured: max |dqpos|
    0.81 against a fresh compile. The safe form is to compile with an envelope that encloses every
    geometry the generator can emit and only move/shrink within it (tools/spikes/).
    """
    global _MODEL, _MODEL_KEY
    key = (course.seed, course.digest)
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = mujoco.MjModel.from_xml_string(_scene_xml(course, None), _mesh_assets())
        _MODEL.opt.timestep = PHYS_DT
        _MODEL_KEY = key
        _COURSE_GEOMS.clear()
        _COURSE_GEOMS.extend(_MODEL.geom(f"{GEOM_PREFIX}{i}").id for i in range(course.n_geoms))
    return _MODEL, _COURSE_GEOMS


class ParkourSim:
    def __init__(self, course: Course, level: float = 0.5, seed: int = 0):
        rng = np.random.default_rng([seed, 0xC0FFEE])
        self.course = course
        self.frictions = sample_frictions(course.segs, level, rng)
        self.level = float(level)
        self.model, geoms = _course_model(course)
        # Sliding friction only; the model's rolling/torsional values stay as authored.
        for gid, mu in zip(geoms, self.frictions):
            self.model.geom_friction[gid, 0] = mu
        self.data = mujoco.MjData(self.model)
        self._pelvis = self.model.body("pelvis").id
        # Scan rays must hit the course, not the robot. mj_ray filters by RENDER GROUP, so the
        # course is emitted into WORLD_GROUP and this mask admits only that group.
        self._ray_mask = np.zeros(6, np.uint8)
        self._ray_mask[WORLD_GROUP] = 1
        self._geomid = np.zeros(1, np.int32)
        self.steps = 0
        self.max_x = START_X
        self._action = np.zeros(ACT_DIM)
        self._seed = seed

    def reset(self, seed: int) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng([seed, 0xBADA55])
        self.data.qpos[0] = START_X
        self.data.qpos[2] = PLINTH_TOP + 0.793
        self.data.qpos[7:] = DEFAULT_ANGLES
        self.data.qpos[:] += rng.uniform(-RESET_NOISE, RESET_NOISE, self.model.nq)
        self.data.qvel[:] += rng.uniform(-RESET_NOISE, RESET_NOISE, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.max_x = START_X
        self._action = np.zeros(ACT_DIM)
        return self._obs()

    def step(self, action, max_steps: int = DEFAULT_MAX_STEPS) -> StepResult:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (ACT_DIM,):
            raise InvalidAction(f"action must be {ACT_DIM} floats, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            # Say which, so a miner debugging a diverging policy is not left guessing.
            bad = int(np.count_nonzero(~np.isfinite(a)))
            raise InvalidAction(f"action must be finite; {bad} of {ACT_DIM} entries are NaN/inf")
        self._action = np.clip(a, -10.0, 10.0)
        target = self._action * ACTION_SCALE + DEFAULT_ANGLES
        for _ in range(FRAME_SKIP):
            self.data.ctrl[:] = (target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.max_x = max(self.max_x, float(self.data.qpos[0]))
        return StepResult(obs=self._obs(), terminal_reason=self._terminal(max_steps))

    @property
    def progress(self) -> float:
        """Fraction of the course covered, in [0, 1]."""
        return float(np.clip((self.max_x - START_X) / (self.course.length - START_X), 0.0, 1.0))

    # -- perception ------------------------------------------------------------------------

    def _yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def _ray_down(self, x: float, y: float, z_from: float) -> float:
        """World height of the first surface below (x, y), or -SCAN_CLIP if there is none."""
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, -1.0]), self._ray_mask, 1, -1, self._geomid)
        return z_from - d if d >= 0 else -SCAN_CLIP

    def _ray_up(self, x: float, y: float, z_from: float) -> float:
        """Clearance above (x, y, z_from) up to SCAN_CLIP; SCAN_CLIP if nothing overhead."""
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, 1.0]), self._ray_mask, 1, -1, self._geomid)
        return SCAN_CLIP if d < 0 else min(d, SCAN_CLIP)

    def _obs(self) -> np.ndarray:
        d, yaw = self.data, self._yaw()
        px, py, pz = (float(v) for v in d.qpos[:3])
        c, s = np.cos(yaw), np.sin(yaw)

        # Body-frame linear/angular velocity and projected gravity.
        rot = np.array(d.xmat[self._pelvis]).reshape(3, 3)
        lin = rot.T @ d.qvel[:3]
        ang = rot.T @ d.qvel[3:6]
        grav = rot.T @ np.array([0.0, 0.0, -1.0])

        # Height scan, in the yaw frame, expressed relative to the pelvis and clipped.
        scan = np.empty(SCAN_NX * SCAN_NY)
        k = 0
        for dx in SCAN_X:
            for dy in SCAN_Y:
                wx, wy = px + c * dx - s * dy, py + s * dx + c * dy
                scan[k] = self._ray_down(wx, wy, pz + RAY_FROM_ABOVE) - pz
                k += 1
        np.clip(scan, -SCAN_CLIP, SCAN_CLIP, out=scan)

        over = np.array([self._ray_up(px + c * dx, py + s * dx, pz + 0.05) for dx in OVERHEAD_X])
        ground = self._ray_down(px, py, pz + RAY_FROM_ABOVE)
        phase = (self.steps * PHYS_DT * FRAME_SKIP % GAIT_PERIOD) / GAIT_PERIOD

        return np.concatenate([
            grav,
            ang * 0.25,
            lin * 2.0,
            d.qpos[7:] - DEFAULT_ANGLES,
            d.qvel[6:] * 0.05,
            self._action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            [np.sin(yaw), np.cos(yaw)],
            [py, (self.course.length - px) / 10.0, np.clip(pz - ground, -SCAN_CLIP, SCAN_CLIP)],
            scan,
            over,
        ]).astype(np.float32)

    # -- gates -----------------------------------------------------------------------------

    def _terminal(self, max_steps: int) -> str | None:
        qpos, qvel = self.data.qpos, self.data.qvel
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            return "physics_glitch"
        if np.max(np.abs(qvel)) > QVEL_GLITCH_LIMIT:
            return "physics_glitch"
        if qpos[0] >= self.course.length:
            return "completed"
        if abs(qpos[1]) > TRACK_HALF_W:
            return "out_of_bounds"
        px, py, pz = (float(v) for v in qpos[:3])
        # Upright means the pelvis's own z axis still points up; xmat[8] is that axis's world z.
        if float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2]) < UPRIGHT_MIN:
            return "fell"
        if pz - self._ray_down(px, py, pz + RAY_FROM_ABOVE) < FALL_CLEARANCE:
            return "fell"
        if self.steps >= max_steps:
            return "timeout"
        return None


def _mesh_assets() -> dict[str, bytes]:
    """MuJoCo resolves meshdir relative to the XML's own path, which from_xml_string does not
    have. Hand it the STLs directly."""
    return {p.name: p.read_bytes() for p in (ASSETS / "meshes").glob("*.STL")}
