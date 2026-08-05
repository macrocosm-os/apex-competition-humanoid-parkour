"""The Humanoid Parkour course: generated per round from the platform's master seed.

The geometry is DRAWN FROM THE ROUND SEED and is not knowable in advance. Every submission in a
round runs the same course — the platform hands every evaluation in the round the same `SEED`
(apex-mvp `SoloRunner._extract_base_seed`, and the per-round seed minted by APEX-97), so identical
resubmissions score identically and there is no seed-fishing. But the course changes when the round
does, so a trajectory optimised offline against last round's layout is worthless in the next one.

This replaces a fixed public layout. That layout made the whole evaluation computable offline
bit-for-bit, which made replaying a memorised joint-target sequence the cheapest route to the top of
the leaderboard — the opposite of the stated success criterion.

What is randomised: the SIZE of every obstacle, within bands. What is not (yet): the ORDER of the
obstacles, and the set of obstacle types. Both are deliberate — see "Bands are capped at today's
validated values" below.

Surface friction still varies per instance and is still not observable (`sample_frictions`), so a
policy has to feel the slip and adapt rather than read a number.

Built as MJCF box geoms on a raised plinth, sized for the Unitree G1 (1.26 m tall, pelvis at
0.784 m). Gaps are real voids in the plinth, so a missed leap is a fall, not a stumble.

    python -m env.course            # print the course for the nominal seed
    python -m env.course 7          # print the course for seed 7
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

PLINTH_TOP = 0.8            # nominal top surface of the track; gaps drop through it
PLINTH_THICK = 0.4
TRACK_HALF_W = 1.2

# Every course geom goes in this MuJoCo render group, and nothing else does. That is what lets
# the height scan ray-cast against the world while ignoring the robot's own body: mj_ray filters
# by group, not by geom. Group 2 is used because the default visualiser shows 0-2, so the course
# still renders without special options; the robot occupies groups 0 (collision) and 1 (visual).
WORLD_GROUP = 2

# Course geoms are named "course_<i>" in emission order, so friction can be set on the compiled
# model rather than baked into the XML.
GEOM_PREFIX = "course_"

# EVERY course is padded to exactly this length. Load-bearing: `progress` is distance over course
# length, `obs[50]` is distance-to-finish, and `max_steps_per_episode` is a fixed step budget. If
# the total length moved with the draw, all three would mean something different every round and
# round-to-round scores would stop being comparable at all.
COURSE_TOTAL_M = 51.0

# The deck may not sink into the floor plane: a slab's top carries PLINTH_THICK of box below it.
DECK_MIN, DECK_MAX = 0.45, 3.0
FINAL_SPRINT_MIN = 2.0      # runway after the last obstacle, so the finish is not a cliff edge

# The steepest short climb the stock G1 walker can still manage: it climbs 15.4 deg but stalls at
# 20.1 deg (docs/design.md). Ramp draws are held at or under this, so the on-ramp stays the "easy
# tier" that a naive walking policy can clear.
ON_RAMP_MAX_GRADE = math.tan(math.radians(15.4))

# Per-instance sliding friction. Normal surfaces vary enough to punish a policy that has
# memorised one contact model; the slick patch is a different regime entirely.
FRICTION_NOMINAL = (0.7, 1.1)
FRICTION_SLICK = (0.12, 0.30)

# Bands are capped at today's validated values, never above them.
#
# Every upper limit here is a number that was checked against this robot: the ramp grade against
# the stock walker's stall angle, the hurdle at 79% of hip height inside the leg's 1.30 m reach,
# the step-up at 31-63 N.m against a 139 N.m knee limit, the duck bar at 1.05 m because a legs-only
# G1 cannot get under much less (all docs/design.md). The `gap`, `beam` and `slick` bands have never
# been audited at all, and nobody has finished the course.
#
# So randomisation draws DOWNWARD from the known-feasible point. Widening any upper limit — or
# permuting obstacle order, where the hazard is the combination rather than the element — needs the
# feasibility audit first. Under randomisation you cannot hand-tune around an infeasible element:
# the generator will happily emit the worst legal combination.
BANDS = {
    "run_up":      (5.0, 7.0),      # today 6.0 — let a walker settle into gait
    "ramp_run":    (2.0, 2.6),      # today 2.0; rise is derived from the grade cap
    "ramp_rise":   (0.35, 0.55),    # today 0.55
    "landing":     (1.4, 1.8),      # today 1.6
    "stairs_up_n": (4, 6),          # today 5
    "stairs_up_rise": (0.16, 0.20),  # today 0.20
    "stairs_up_run":  (0.30, 0.36),  # today 0.32
    "leap":        (0.80, 1.00),    # today 1.00 — NOT audited, so never above today
    "drop":        (0.50, 0.60),    # today 0.60
    "hurdle":      (0.55, 0.62),    # today 0.62 = 79% of hip height
    "step_up":     (0.45, 0.55),    # today 0.55
    "duck_bar":    (1.02, 1.10),    # today 1.05; below ~1.0 no legs-only G1 can pass
    "beam_len":    (3.0, 3.5),      # today 3.5 — NOT audited
    "beam_half_w": (0.16, 0.22),    # today 0.16; wider is easier, so this one may go up
    "slick_len":   (2.5, 3.5),      # today 3.0 — NOT audited
    "stairs_dn_n": (5, 6),          # today 6
    "stairs_dn_rise": (0.16, 0.18),  # today 0.18
    "stairs_dn_run":  (0.32, 0.36),  # today 0.34
    "filler":      (1.6, 2.8),      # the flats between obstacles
}

MAX_REDRAWS = 200           # a generator that cannot satisfy its own guards must fail loudly

COLOR = {  # by maneuver, so a render is readable at a glance
    "flat": ".55 .57 .60 1", "ramp": ".45 .80 .55 1",
    "stairs_up": ".30 .65 .45 1", "stairs_dn": ".22 .50 .38 1",
    "step_up": ".85 .45 .20 1", "drop_down": ".70 .35 .18 1",
    "leap": ".20 .20 .24 1", "hurdle": ".80 .25 .35 1",
    "duck": ".55 .30 .75 1", "beam": ".95 .75 .20 1", "slick": ".35 .70 .95 1",
}


@dataclass
class Seg:
    kind: str
    length: float
    boxes: list = field(default_factory=list)   # (cx, cy, cz, sx, sy, sz, color[, pitch])


@dataclass(frozen=True)
class Course:
    """One generated course. `digest` identifies the geometry for audit without revealing it."""
    seed: int
    segs: list
    length: float
    final_deck: float

    @property
    def n_geoms(self) -> int:
        return sum(len(s.boxes) for s in self.segs)

    @property
    def digest(self) -> str:
        """Stable hash of the emitted geometry. Recorded in the round's result metadata so a course
        is reproducible and a dispute is answerable, without publishing the layout mid-round."""
        h = hashlib.sha256()
        for s in self.segs:
            h.update(s.kind.encode())
            for b in s.boxes:
                h.update(np.asarray(b[:6], dtype=np.float64).tobytes())
                h.update(f"{b[7]:.6f}".encode() if len(b) > 7 else b"0")
        return h.hexdigest()[:16]


def _slab(x0, length, top, color, half_w=TRACK_HALF_W):
    """A walkable slab whose upper surface sits at `top`."""
    return (x0 + length / 2, 0.0, top - PLINTH_THICK / 2, length / 2, half_w, PLINTH_THICK / 2, color)


def _ramp(x0, length, top0, rise, color):
    """A slab rotated about y so its top face is an incline climbing `rise` over `length`."""
    ang = math.atan2(rise, length)
    return (x0 + length / 2, 0.0, top0 + rise / 2 - (PLINTH_THICK / 2) * math.cos(ang),
            math.hypot(length, rise) / 2, TRACK_HALF_W, PLINTH_THICK / 2, color, -ang)


def _draw(rng, key):
    lo, hi = BANDS[key]
    if isinstance(lo, int) and isinstance(hi, int):
        return int(rng.integers(lo, hi + 1))
    return float(rng.uniform(lo, hi))


def _build(rng) -> Course | None:
    """One attempt. Returns None if the draw violates a guard, so the caller can redraw."""
    segs: list[Seg] = []
    x, top = 0.0, PLINTH_TOP
    decks = [top]

    def flat(length, kind="flat"):
        nonlocal x
        segs.append(Seg(kind, length, [_slab(x, length, top, kind)]))
        x += length

    def stairs(n, rise, run, kind):
        nonlocal x, top
        s = Seg(kind, n * run)
        step = rise if kind == "stairs_up" else -rise
        for i in range(n):
            s.boxes.append(_slab(x + i * run, run, top + (i + 1) * step, kind))
        segs.append(s)
        x += n * run
        top += n * step
        decks.append(top)

    # ON-RAMP: a naive walking policy should clear this and nothing beyond it. Because the course
    # is linear and scored on progress, this section IS the easy tier — no separate tiers needed.
    flat(_draw(rng, "run_up"))
    ramp_run = _draw(rng, "ramp_run")
    # Draw the rise, then cap it at the grade the stock walker can still climb.
    ramp_rise = min(_draw(rng, "ramp_rise"), ramp_run * ON_RAMP_MAX_GRADE)
    segs.append(Seg("ramp_up", ramp_run, [_ramp(x, ramp_run, top, ramp_rise, "ramp")]))
    x += ramp_run
    top += ramp_rise
    decks.append(top)
    flat(_draw(rng, "landing"))                            # to set up for the edge
    top -= ramp_rise                                       # sheer drop back to the plinth
    segs.append(Seg("drop_down", 0.0))
    decks.append(top)
    flat(_draw(rng, "filler"))

    # THE COURSE PROPER
    flat(_draw(rng, "filler"))
    stairs(_draw(rng, "stairs_up_n"), _draw(rng, "stairs_up_rise"), _draw(rng, "stairs_up_run"),
           "stairs_up")
    flat(_draw(rng, "filler"))
    leap = _draw(rng, "leap")
    segs.append(Seg("leap", leap))                         # a real void: no slab at all
    x += leap
    flat(_draw(rng, "filler"))
    top -= _draw(rng, "drop")
    segs.append(Seg("drop_down", 0.0))
    decks.append(top)
    flat(_draw(rng, "filler"))

    # A barrier the robot must step OVER, not vault: it has no arms (env/sim.py). Held at or under
    # 79% of hip height, well inside the leg's 1.30 m kinematic reach.
    hurdle_h = _draw(rng, "hurdle")
    segs.append(Seg("hurdle", 1.0, [_slab(x, 1.0, top, "flat"),
                                    (x + 0.5, 0.0, top + hurdle_h / 2, 0.09, TRACK_HALF_W,
                                     hurdle_h / 2, "hurdle")]))
    x += 1.0
    flat(_draw(rng, "filler"))

    # A step UP, not a climb — again, no arms to pull with. A torque-feasible single-leg press.
    plat = _draw(rng, "step_up")
    segs.append(Seg("step_up", 2.2, [_slab(x, 2.2, top + plat, "step_up")]))
    x += 2.2
    top += plat
    decks.append(top)
    flat(_draw(rng, "filler"))
    top -= plat                                            # and back down
    decks.append(top)
    flat(_draw(rng, "filler"))

    bar = _draw(rng, "duck_bar")
    s = Seg("duck", 2.0, [_slab(x, 2.0, top, "flat"),      # overhead bar on posts
                          (x + 1.0, 0.0, top + bar + 0.08, 0.5, TRACK_HALF_W, 0.08, "duck")])
    for sy in (-1.0, 1.0):
        s.boxes.append((x + 1.0, sy * (TRACK_HALF_W - 0.06), top + bar / 2,
                        0.06, 0.06, bar / 2, "duck"))
    segs.append(s)
    x += 2.0
    flat(_draw(rng, "filler"))

    beam_len = _draw(rng, "beam_len")
    segs.append(Seg("beam", beam_len,
                    [_slab(x, beam_len, top, "beam", half_w=_draw(rng, "beam_half_w"))]))
    x += beam_len
    flat(_draw(rng, "filler"))
    flat(_draw(rng, "slick_len"), kind="slick")             # same geometry, low friction
    stairs(_draw(rng, "stairs_dn_n"), _draw(rng, "stairs_dn_rise"), _draw(rng, "stairs_dn_run"),
           "stairs_dn")

    # Pad the final sprint so every course is exactly COURSE_TOTAL_M long.
    sprint = COURSE_TOTAL_M - x
    if sprint < FINAL_SPRINT_MIN:
        return None
    if not all(DECK_MIN <= d <= DECK_MAX for d in decks):
        return None
    flat(sprint)
    return Course(seed=-1, segs=segs, length=x, final_deck=top)


def generate_course(seed: int) -> Course:
    """Deterministically generate one course. Same seed -> same course, bit for bit.

    The referee calls this once per round with the platform's master seed, so the layout is fixed
    for everyone inside a round and different between rounds.
    """
    for attempt in range(MAX_REDRAWS):
        # Fold the attempt into the stream so a rejected draw does not just repeat.
        c = _build(np.random.default_rng([seed, attempt, 0xC0FFEE]))
        if c is not None:
            return Course(seed=seed, segs=c.segs, length=c.length, final_deck=c.final_deck)
    raise RuntimeError(
        f"course generation failed for seed {seed} after {MAX_REDRAWS} redraws — the BANDS and the "
        f"COURSE_TOTAL_M / DECK_MIN / DECK_MAX guards are mutually unsatisfiable, which is a bug in "
        f"this module, not bad luck"
    )


def course_xml_fragment(segs, frictions=None):
    """MJCF for the course. `frictions` is one sliding-friction value per emitted geom, in the
    same order this function walks them — see `sample_frictions`."""
    out, i = [], 0
    for s in segs:
        for b in s.boxes:
            cx, cy, cz, sx, sy, sz, ck = b[:7]
            euler = f' euler="0 {b[7]:.4f} 0"' if len(b) > 7 else ""
            mu = 1.0 if frictions is None else frictions[i]
            # Named so the sim can set friction on the compiled model instead of recompiling
            # it per instance. Emission order is the contract with `sample_frictions`.
            out.append(f'    <geom name="{GEOM_PREFIX}{i}" type="box" '
                       f'pos="{cx:.3f} {cy:.3f} {cz:.3f}" '
                       f'size="{sx:.3f} {sy:.3f} {sz:.3f}"{euler} condim="3" group="{WORLD_GROUP}" '
                       f'friction="{mu:.4f} .1 .1" rgba="{COLOR[ck]}"/>')
            i += 1
    return "\n".join(out)


def sample_frictions(segs, level: float, rng: np.random.Generator) -> list[float]:
    """One sliding friction per geom, in `course_xml_fragment` order.

    `level` in [0, 1] slides the whole course from the grippy end of its range to the slippery
    end; `rng` adds a little per-geom jitter on top so no two slabs are exactly alike. Slick
    slabs use their own, much lower range. The split matters: `level` is what the evaluation
    suite STRATIFIES over (env/sim.py), so a fixed set of instances covers the whole friction
    continuum evenly instead of sampling it at random.
    """
    out = []
    for s in segs:
        lo, hi = FRICTION_SLICK if s.kind == "slick" else FRICTION_NOMINAL
        base = hi - (hi - lo) * float(level)
        for _ in s.boxes:
            jitter = (hi - lo) * 0.08 * float(rng.uniform(-1.0, 1.0))
            # Rounded to 4 dp deliberately. These values used to reach MuJoCo through the XML,
            # which serialised them at %.4f; they are now written straight into geom_friction.
            # Quantising here makes the two paths agree exactly instead of differing by the
            # serialisation rounding, which was worth 0.15% of raw_score -- inside the 1%
            # takeover margin, so not something to leave to a format string.
            out.append(round(float(np.clip(base + jitter, lo, hi)), 4))
    return out


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    course = generate_course(seed)
    x = 0.0
    print(f"seed {seed}   digest {course.digest}")
    print(f"{'segment':12} {'x start':>8} {'length':>7}")
    for s in course.segs:
        print(f"{s.kind:12} {x:>8.2f} {s.length:>7.2f}")
        x += s.length
    zs = [b[2] + b[5] for s in course.segs for b in s.boxes if b[6] not in ("hurdle", "duck")]
    print(f"\ntotal {course.length:.1f} m, final deck {course.final_deck:.2f} m, "
          f"vertical range {max(zs) - min(zs):.2f} m, {course.n_geoms} geoms")
