"""The course's sliding friction must reach the solver, not just the model.

Writing `geom_friction` on the course is necessary but not sufficient. MuJoCo mixes contact
parameters from both geoms in a pair, and for friction the mix is the element-wise MAXIMUM when
the two geoms have equal `geom_priority`. `g1_12dof.xml` sets no geom friction, so the robot's
feet take MuJoCo's default 1.0 -- above every mu this course draws. At equal priority every foot
contact would therefore solve at 1.0 and the band would never reach the solver, slick patch
included. `_shared_model` raises the course geoms' priority to prevent that.

These assert at CONTACT level on purpose. A score-based check cannot tell "friction was applied"
apart from "friction was mixed away and the policy happens to be robust", so it is not evidence
either way.

    python -m pytest tests/test_friction_reaches_contacts.py
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import env.course as course
from env.sim import InstanceParams, ParkourSim


def _geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def _settle(sim: ParkourSim, steps: int = 60) -> None:
    """Hold the default pose long enough for foot contacts to form."""
    sim.reset()
    for _ in range(steps):
        sim.step(np.zeros(12))


def _course_contacts(sim: ParkourSim) -> list[tuple[float, float]]:
    """(geom mu, solved contact mu) for every contact involving a course geom."""
    out = []
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        for gid in (con.geom1, con.geom2):
            if _geom_name(sim.model, gid).startswith(course.GEOM_PREFIX):
                out.append((float(sim.model.geom_friction[gid, 0]),
                            float(con.friction[0])))
    return out


def test_course_geoms_outrank_the_robot():
    """Course geoms must carry a higher contact priority than the robot's."""
    sim = ParkourSim(InstanceParams(seed=1, friction_level=1.0,
                                    wind_speed=0.0, wind_dir=0.0))
    course_prio, robot_prio = set(), set()
    for gid in range(sim.model.ngeom):
        target = course_prio if _geom_name(sim.model, gid).startswith(course.GEOM_PREFIX) \
            else robot_prio
        target.add(int(sim.model.geom_priority[gid]))

    assert course_prio, "no course geoms found"
    assert min(course_prio) > max(robot_prio), (
        f"course priority {course_prio} must exceed everything else {robot_prio}; "
        "at equal priority MuJoCo takes max() of the two frictions and the course loses"
    )


def test_solved_contact_friction_matches_the_course():
    """The mu MuJoCo actually solves with must be the mu the course asked for."""
    sim = ParkourSim(InstanceParams(seed=1, friction_level=1.0,
                                    wind_speed=0.0, wind_dir=0.0))
    _settle(sim)
    contacts = _course_contacts(sim)

    assert contacts, "no foot/course contacts formed; the test cannot conclude anything"
    for geom_mu, contact_mu in contacts:
        assert contact_mu == pytest.approx(geom_mu, abs=1e-6), (
            f"course geom asked for mu {geom_mu:.4f} but the solver used {contact_mu:.4f}"
        )


def test_contact_friction_is_not_pinned_to_the_default():
    """Lowering the band must move the solved friction, not just the model field.

    A naive "is geom_friction set?" check passes even when every solved contact sits at MuJoCo's
    1.0 default, so this pins the failure mode directly rather than inspecting the model alone.
    """
    nominal, slick = course.FRICTION_NOMINAL, course.FRICTION_SLICK
    seen = []
    try:
        for band in ((0.50, 1.25), (0.10, 0.20)):
            course.FRICTION_NOMINAL, course.FRICTION_SLICK = band, (band[0] / 4, band[1] / 4)
            sim = ParkourSim(InstanceParams(seed=1, friction_level=1.0,
                                            wind_speed=0.0, wind_dir=0.0))
            _settle(sim)
            contacts = _course_contacts(sim)
            assert contacts, f"no foot/course contacts formed for band {band}"
            seen.append({round(mu, 4) for _, mu in contacts})
    finally:
        course.FRICTION_NOMINAL, course.FRICTION_SLICK = nominal, slick

    high, low = seen
    assert high != low, (
        f"solved contact friction did not change when the band did ({high} vs {low}) -- "
        "the course's friction is not reaching the solver"
    )
    assert low != {1.0}, "solved contact friction is pinned at MuJoCo's 1.0 default"
    assert max(low) < min(high), (
        f"a lower band must produce lower solved friction, got {low} vs {high}"
    )


def test_slick_patch_is_slicker_than_the_nominal_course():
    """The slick segment must actually reach the solver as a distinct, lower-friction regime."""
    sim = ParkourSim(InstanceParams(seed=1, friction_level=1.0,
                                    wind_speed=0.0, wind_dir=0.0))
    idx, slick_ids = 0, set()
    for seg in course.SEGMENTS:
        for _ in seg.boxes:
            if seg.kind == "slick":
                slick_ids.add(idx)
            idx += 1

    assert slick_ids, "course has no slick segment"
    slick_mu = [float(sim.model.geom_friction[sim.model.geom(f"{course.GEOM_PREFIX}{i}").id, 0])
                for i in sorted(slick_ids)]
    other_mu = [float(sim.model.geom_friction[sim.model.geom(f"{course.GEOM_PREFIX}{i}").id, 0])
                for i in range(idx) if i not in slick_ids]

    assert max(slick_mu) < min(other_mu), (
        f"slick patch mu {slick_mu} must sit below the nominal course {min(other_mu):.4f}"
    )
    assert max(slick_mu) < 1.0, "slick patch must not be at MuJoCo's default friction"
