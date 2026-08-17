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

The downward channels ([51] and the scan) report WALKABLE SURFACES only; overhead structures such
as the duck bar appear in [97:104] and nowhere else. See `course.OVERHEAD_GROUP`.

Action (float32, ACT_DIM = 12): joint position targets as offsets from the default pose,
scaled by ACTION_SCALE and clipped to the joint limits.

Surface friction and wind vary between instances and are NOT in the observation. Adapting to
conditions you cannot see is the point (env/course.py). Both are drawn at random from the round
seed, so the suite differs every round — see `instance_spec`.

Termination gates (each maps to a terminal_reason the miner sees post-round):
    completed       pelvis past the finish line
    fell            pelvis under FALL_CLEARANCE above the walkable surface below it, or torso
                    past ~66 deg
    out_of_bounds   |y| > TRACK_HALF_W (no walking around the course)
    physics_glitch  NaN/Inf state or |qvel| > 100 (glitch-surfing scores 0)
    timeout         max_steps control steps elapsed
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

from .course import (COURSE_LENGTH, GEOM_PREFIX, OVERHEAD_GROUP, PLINTH_TOP, SEGMENTS,
                     TRACK_HALF_W, WORLD_GROUP, course_xml_fragment, sample_frictions)

ASSETS = pathlib.Path(__file__).parent / "assets"

ACT_DIM = 12

# Opaque per-episode policy memory, threaded by the player between /act calls and zeroed on
# /reset. Recurrence is not a luxury here: friction and wind are randomised and NEITHER is
# observable, so a policy can only adapt by remembering that it just slipped or got pushed. A
# feed-forward submission simply returns zeros and ignores it. The stock walker is itself an LSTM.
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

# Wind, via MuJoCo's inertia-box fluid model: opt.wind is subtracted from each body's linear
# velocity and quadratic drag follows, so it only bites with opt.density > 0. Air at 20 C.
# This robot's equivalent-inertia boxes sum to 0.297 m^2 of frontal area, so drag is
# 0.179 N per (m/s)^2 head-on -- 35.1 N at WIND_MAX_MS, 11.1% of the G1's 315 N weight.
# WIND_MAX_MS is Beaufort 7 measured AT THE ROBOT, which is stronger weather than it sounds:
# wind at 1 m is 50-70% of the 10 m figure forecasts quote.
#
# 14 is set below a measured ceiling, not guessed. Drag scales as v^2 against a fixed weight, so
# the mu needed just to hold station is 0.179 v^2 / 315: at 22 m/s that is 0.275, and nothing
# completes the course at 22 in any direction even with maximum grip. Hurricane force (32.7 m/s)
# needs mu 0.608 and a 36.5 cm centre-of-pressure shift against a ~9 cm foot -- it is outside the
# physics, not merely hard, and no policy can be trained into it. The last speed that still
# completes is 20; 14 leaves headroom for the friction band to be the binding constraint.
AIR_DENSITY = 1.204
WIND_MAX_MS = 14.0


class InvalidAction(ValueError):
    """The action was not a finite ACT_DIM vector."""


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    terminal_reason: str | None  # None while the episode is still running


def _scene_xml(frictions: list[float] | None) -> str:
    """The robot model with the course spliced into its worldbody."""
    robot = (ASSETS / "g1_12dof.xml").read_text()
    floor = (f'    <geom name="floor" type="plane" size="80 20 0.1" pos="30 0 0" '
             f'condim="3" group="{WORLD_GROUP}" rgba=".18 .19 .22 1"/>\n')
    start = (f'    <geom type="box" pos="{START_X - 0.7:.3f} 0 {PLINTH_TOP - 0.2:.3f}" '
             f'size="1.5 {TRACK_HALF_W} 0.2" condim="3" group="{WORLD_GROUP}" '
             f'friction="1 .1 .1" rgba=".45 .47 .5 1"/>\n')
    body = floor + start + course_xml_fragment(SEGMENTS, frictions)
    return robot.replace("</worldbody>", body + "\n  </worldbody>")


@dataclass(frozen=True)
class InstanceParams:
    """The randomised conditions of one evaluation instance."""

    seed: int                # per-instance episode seed: friction jitter and reset noise
    friction_level: float    # 0 = grippy end of the band, 1 = slippery end
    wind_speed: float        # m/s
    wind_dir: float          # radians; the direction the wind blows FROM, about +x

    @property
    def wind(self) -> tuple[float, float, float]:
        """World-frame air velocity for opt.wind. Horizontal only, as ground-level wind is.

        dir 0 is a headwind for a robot travelling +x (air moves -x); dir pi/2 blows from +y and
        pushes it toward -y.
        """
        return (-self.wind_speed * float(np.cos(self.wind_dir)),
                -self.wind_speed * float(np.sin(self.wind_dir)), 0.0)


def instance_spec(i: int, n: int, seed: int,
                  wind_max: float = WIND_MAX_MS) -> InstanceParams:
    """The conditions of evaluation instance `i` of `n`, drawn from the round `seed`.

    Friction and wind are drawn at RANDOM per instance rather than taken from a fixed
    stratified sweep, so the suite differs every round. The course geometry is static and
    public, so a fixed suite was computable offline bit-for-bit and the cheapest route to the
    top was memorising 24 known instances; a per-round draw makes that worthless.

    The cost is score noise, and score noise is what sets the takeover margin — round-to-round
    variance is no longer zero. See docs/design.md, "Randomised conditions on a fixed course".

    `seed` has no default on purpose: forgetting it would silently freeze the suite, which is the
    failure this change exists to remove. `n` is deliberately NOT an input to the draw — instance
    `i` gets the same conditions whatever the suite size, so raising `num_instances` extends the
    suite instead of reshuffling it.
    """
    rng = np.random.default_rng([seed, i, 0x5EED])
    return InstanceParams(
        # Derived from the round seed too, so reset noise no longer fingerprints the instance.
        seed=int(rng.integers(1 << 31)),
        friction_level=float(rng.uniform(0.0, 1.0)),
        wind_speed=float(rng.uniform(0.0, wind_max)),
        wind_dir=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


_MODEL: mujoco.MjModel | None = None
_COURSE_GEOMS: list[int] = []


def _shared_model() -> tuple[mujoco.MjModel, list[int]]:
    """Compile the scene once and reuse it for every instance.

    Compiling costs ~1.1 s and a few hundred MB, because the G1's collision geometry is 27 STL
    meshes that MuJoCo converts to convex hulls. Doing that per instance was ~26 s of a 67 s
    evaluation and pushed the referee to 1.1 GB against a 1.5 GiB limit.

    Nothing instance-specific is baked into the model any more: friction and wind are what vary,
    and `geom_friction` / `opt.wind` are runtime fields, so both are written per instance in
    __init__. Instances run strictly sequentially and each gets a fresh MjData, so sharing the
    model is safe. Verified bit-identical to per-instance compilation.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_string(_scene_xml(None), _mesh_assets())
        _MODEL.opt.timestep = PHYS_DT
        # Constant, so it belongs here rather than per instance. Enables the fluid model that
        # opt.wind acts through; it also adds still-air drag, which is why the baseline moved.
        _MODEL.opt.density = AIR_DENSITY
        n = sum(len(s.boxes) for s in SEGMENTS)
        _COURSE_GEOMS.extend(_MODEL.geom(f"{GEOM_PREFIX}{i}").id for i in range(n))
        # Make the course's friction authoritative for foot contacts. Writing geom_friction is
        # necessary but not sufficient: MuJoCo mixes contact parameters from BOTH geoms in a pair,
        # and for friction the mix is the element-wise MAXIMUM whenever the two have equal
        # priority. g1_12dof.xml declares no geom friction, so the robot's feet sit at MuJoCo's
        # default of 1.0 -- above every mu this course draws, so max() would take the foot's value
        # and the course's band would not reach the solver at all. Raising priority on the course
        # side makes its contact parameters win outright, which is MuJoCo's documented mechanism
        # for exactly this case. Guarded by tests/test_friction_reaches_contacts.py, which asserts
        # on the solved contact friction rather than on a score -- a score cannot distinguish a
        # band that applied from one that was mixed away.
        #
        # Constant per geom, so it belongs here; only geom_friction varies per instance.
        for gid in _COURSE_GEOMS:
            _MODEL.geom_priority[gid] = 1
    return _MODEL, _COURSE_GEOMS


class ParkourSim:
    def __init__(self, params: InstanceParams):
        self.params = params
        rng = np.random.default_rng([params.seed, 0xC0FFEE])
        self.frictions = sample_frictions(SEGMENTS, self.params.friction_level, rng)
        self.level = self.params.friction_level
        self.model, geoms = _shared_model()
        # Sliding friction only; the model's rolling/torsional values stay as authored.
        for gid, mu in zip(geoms, self.frictions):
            self.model.geom_friction[gid, 0] = mu
        self.model.opt.wind[:] = self.params.wind
        self.data = mujoco.MjData(self.model)
        self._pelvis = self.model.body("pelvis").id
        # Rays must hit the course, not the robot. mj_ray filters by RENDER GROUP, so the course
        # is emitted into WORLD_GROUP and these masks admit course geoms only. DOWNWARD rays --
        # the fall gate and the height scan -- admit walkable surfaces alone, so an overhead
        # structure can never answer "what is the ground here?". UPWARD rays admit both, which is
        # how the duck-under stays visible.
        self._ray_mask = np.zeros(6, np.uint8)
        self._ray_mask[WORLD_GROUP] = 1
        self._up_mask = np.zeros(6, np.uint8)
        self._up_mask[WORLD_GROUP] = 1
        self._up_mask[OVERHEAD_GROUP] = 1
        self._geomid = np.zeros(1, np.int32)
        self.steps = 0
        self.max_x = START_X
        self._action = np.zeros(ACT_DIM)
        self._seed = params.seed

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng([self._seed, 0xBADA55])
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
        return float(np.clip((self.max_x - START_X) / (COURSE_LENGTH - START_X), 0.0, 1.0))

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
                          np.array([0.0, 0.0, 1.0]), self._up_mask, 1, -1, self._geomid)
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
            [py, (COURSE_LENGTH - px) / 10.0, np.clip(pz - ground, -SCAN_CLIP, SCAN_CLIP)],
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
        if qpos[0] >= COURSE_LENGTH:
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
