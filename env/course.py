"""The Humanoid Parkour course: one fixed layout, per-instance surface friction.

The geometry is STATIC and public — every instance in every round runs this exact course.
Randomising the layout was considered and dropped: layout noise adds far more score variance
than it removes memorisation risk (docs/design.md, "Randomised conditions on a fixed course").
What DOES vary per instance is the sliding friction of every surface, drawn at random from the
round seed, and it is deliberately NOT observable — a policy has to feel the slip and adapt
rather than read a number.

Built as MJCF box geoms on a raised plinth, sized for the Unitree G1 (1.26 m tall, pelvis at
0.784 m). Gaps are real voids in the plinth, so a missed leap is a fall, not a stumble.

    python -m env.course        # print the layout
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PLINTH_TOP = 0.8            # top surface of the track; gaps drop through it
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

# Kinds that are structures to pass UNDER rather than surfaces to stand on, and the group they
# are emitted into. mj_ray stops at the first surface it meets and filters by render group, so
# giving these their own group is what lets a DOWNWARD ray ignore them: without it the duck bar's
# top face answers "what is the ground here?" with a slab 1.2 m above the deck, to the fall gate
# and to the height scan alike. Upward rays still admit this group -- overhead clearance is how
# the duck-under is meant to be visible (README). Group is a ray/render filter, NOT a collision
# filter: condim/contype/conaffinity are untouched, so the bar is exactly as solid as before.
OVERHEAD_KINDS = frozenset({"duck"})
OVERHEAD_GROUP = 3

# On-ramp shape, measured against the stock G1 walker (see docs/design.md): it climbs
# 15.4 deg but stalls at 20.1 deg, so this is the steepest short climb a naive policy can still
# manage. Drop height barely matters — 0.20 m and 0.55 m both end its run — so we take the full
# drop for the spectacle.
ON_RAMP_RISE, ON_RAMP_RUN, ON_RAMP_DROP = 0.55, 2.0, 0.55

# Overhead bar height. The G1 stands 1.26 m, so this forces a ~0.2 m squat-walk. It is NOT a
# crawl: with 12 leg DoF and a welded upper body the robot cannot get its head under anything
# much lower, and a segment no embodiment can clear is just a wall.
DUCK_BAR_Z = 1.05

# Per-instance sliding friction, drawn at random (env/sim.instance_spec). Bands are NARROW and
# LOW on purpose. `friction_level` is uniform on [0, 1] and maps linearly across the band, so a
# wide band spends most of its draws nowhere near the slippery end: [0.50, 1.25] put the average
# instance at mu 0.875, and lowering only the floor barely moved a round. Narrowing is what makes
# every instance a friction problem. 0.50 is wet concrete, 0.35 is wet tile; the slick patch is a
# different regime again, down to polished ice.
#
# BOTH ends are pinned by measurement, and the window between them is genuinely small.
#
# The floor is bounded from below by the course. The on-ramp is atan(0.55/2.0) = 15.38 deg, so
# walking up it needs mu >= tan(15.38 deg) = 0.275 no matter how good the policy is -- and a
# walking biped needs real margin over that static limit, because it has to push off and brake as
# well as stand. Measured: every band capped below 0.40 yields 0 completions in 120 instances and
# runs pile up on the ramp at 5-8 m of 51.14 m. Do not take the floor below 0.35.
#
# The ceiling is bounded from above by the point of the exercise: it sets how many instances are
# grippy enough to be easy. Dropping it from 0.55 to 0.50 takes mean mu from 0.442 to 0.419 and
# the leader's completion rate from 6.7% to 4.2% of instances. 0.48 and below falls off a cliff
# to under 1%, with whole round seeds yielding no completion at all -- which spikes round-to-round
# variance rather than measuring skill. 0.50 is the last ceiling that stays robustly completable.
FRICTION_NOMINAL = (0.35, 0.50)
FRICTION_SLICK = (0.08, 0.14)

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


def _slab(x0, length, top, color, half_w=TRACK_HALF_W):
    """A walkable slab whose upper surface sits at `top`."""
    return (x0 + length / 2, 0.0, top - PLINTH_THICK / 2, length / 2, half_w, PLINTH_THICK / 2, color)


def _ramp(x0, length, top0, rise, color):
    """A slab rotated about y so its top face is an incline climbing `rise` over `length`."""
    ang = math.atan2(rise, length)
    return (x0 + length / 2, 0.0, top0 + rise / 2 - (PLINTH_THICK / 2) * math.cos(ang),
            math.hypot(length, rise) / 2, TRACK_HALF_W, PLINTH_THICK / 2, color, -ang)


def build_course():
    """Return (segments, total_length, final_deck_height)."""
    segs: list[Seg] = []
    x, top = 0.0, PLINTH_TOP

    def flat(length, kind="flat"):
        nonlocal x
        segs.append(Seg(kind, length, [_slab(x, length, top, kind)]))
        x += length

    def stairs(n, rise, run, kind):
        nonlocal x, top
        s = Seg(kind, n * run)
        for i in range(n):
            step = rise if kind == "stairs_up" else -rise
            s.boxes.append(_slab(x + i * run, run, top + (i + 1) * step, kind))
        segs.append(s); x += n * run; top += n * (rise if kind == "stairs_up" else -rise)

    # ON-RAMP: a naive walking policy should clear this and nothing beyond it. Because the course
    # is linear and scored on progress, this section IS the easy tier — no separate tiers needed.
    flat(6.0)                                              # run-up: let a walker settle into gait
    segs.append(Seg("ramp_up", ON_RAMP_RUN, [_ramp(x, ON_RAMP_RUN, top, ON_RAMP_RISE, "ramp")]))
    x += ON_RAMP_RUN; top += ON_RAMP_RISE
    flat(1.6)                                              # landing, to set up for the edge
    top -= ON_RAMP_DROP                                    # sheer drop, no ramp down
    segs.append(Seg("drop_down", 0.0))
    flat(2.4)

    # THE COURSE PROPER
    flat(4.0)
    stairs(5, 0.20, 0.32, "stairs_up")
    flat(1.6)
    segs.append(Seg("leap", 1.0)); x += 1.0                # a real void: no slab at all
    flat(2.2)
    top -= 0.6                                             # drop-down
    segs.append(Seg("drop_down", 0.0))
    flat(2.4)

    # A 0.62 m barrier the robot must step OVER, not vault: it has no arms (env/sim.py). That is
    # 79% of hip height, well inside the leg's 1.30 m kinematic reach.
    s = Seg("hurdle", 1.0, [_slab(x, 1.0, top, "flat"),
                            (x + 0.5, 0.0, top + 0.31, 0.09, TRACK_HALF_W, 0.31, "hurdle")])
    segs.append(s); x += 1.0
    flat(2.0)

    # A 0.55 m step UP, not a climb — again, no arms to pull with. Needs ~31-63 N.m at the knee
    # against a 139 N.m limit, so it is a torque-feasible single-leg press.
    plat = 0.55
    segs.append(Seg("step_up", 2.2, [_slab(x, 2.2, top + plat, "step_up")]))
    x += 2.2; top += plat
    flat(1.2)
    top -= plat                                            # and back down
    flat(2.0)

    s = Seg("duck", 2.0, [_slab(x, 2.0, top, "flat"),      # overhead bar on posts
                          (x + 1.0, 0.0, top + DUCK_BAR_Z + 0.08, 0.5, TRACK_HALF_W, 0.08, "duck")])
    for sy in (-1.0, 1.0):
        s.boxes.append((x + 1.0, sy * (TRACK_HALF_W - 0.06), top + DUCK_BAR_Z / 2,
                        0.06, 0.06, DUCK_BAR_Z / 2, "duck"))
    segs.append(s); x += 2.0
    flat(1.6)

    segs.append(Seg("beam", 3.5, [_slab(x, 3.5, top, "beam", half_w=0.16)]))
    x += 3.5
    flat(1.8)
    flat(3.0, kind="slick")                                # same geometry, low friction
    stairs(6, 0.18, 0.34, "stairs_dn")
    flat(4.0)                                              # final sprint to the line
    return segs, x, top


SEGMENTS, COURSE_LENGTH, FINAL_DECK = build_course()


def course_xml_fragment(segs, frictions=None):
    """MJCF for the course. `frictions` is one sliding-friction value per emitted geom, in the
    same order this function walks them — see `sample_frictions`."""
    out, i = [], 0
    for s in segs:
        for b in s.boxes:
            cx, cy, cz, sx, sy, sz, ck = b[:7]
            euler = f' euler="0 {b[7]:.4f} 0"' if len(b) > 7 else ""
            mu = 1.0 if frictions is None else frictions[i]
            grp = OVERHEAD_GROUP if ck in OVERHEAD_KINDS else WORLD_GROUP
            # Named so the sim can set friction on the compiled model instead of recompiling
            # it per instance. Emission order is the contract with `sample_frictions`.
            out.append(f'    <geom name="{GEOM_PREFIX}{i}" type="box" '
                       f'pos="{cx:.3f} {cy:.3f} {cz:.3f}" '
                       f'size="{sx:.3f} {sy:.3f} {sz:.3f}"{euler} condim="3" group="{grp}" '
                       f'friction="{mu:.4f} .1 .1" rgba="{COLOR[ck]}"/>')
            i += 1
    return "\n".join(out)


def nominal_mu(level: float) -> float:
    """The course-wide sliding friction `level` maps to, before per-geom jitter. For reporting."""
    lo, hi = FRICTION_NOMINAL
    return hi - (hi - lo) * float(level)


def sample_frictions(segs, level: float, rng: np.random.Generator) -> list[float]:
    """One sliding friction per geom, in `course_xml_fragment` order.

    `level` in [0, 1] slides the whole course from the grippy end of its range to the slippery
    end; `rng` adds a little per-geom jitter on top so no two slabs are exactly alike. Slick
    slabs use their own, much lower range. One `level` per instance rather than one per geom:
    a real run happens on one surface family, not on a patchwork that changes every slab.
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
    x = 0.0
    print(f"{'segment':12} {'x start':>8} {'length':>7}")
    for s in SEGMENTS:
        print(f"{s.kind:12} {x:>8.2f} {s.length:>7.2f}")
        x += s.length
    zs = [b[2] + b[5] for s in SEGMENTS for b in s.boxes if b[6] not in ("hurdle", "duck")]
    print(f"\ntotal {COURSE_LENGTH:.1f} m, final deck {FINAL_DECK:.2f} m, "
          f"vertical range {max(zs) - min(zs):.2f} m, {sum(len(s.boxes) for s in SEGMENTS)} geoms")
