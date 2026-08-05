"""Render the course: stills, a flythrough, and a run by an actual policy.

    python tools/preview.py                              # stills + flythrough
    python tools/preview.py --run baseline/baseline.onnx  # + a scored run, filmed

Everything renders against the same vendored G1 and the same `ParkourSim` the referee uses, so
what you see is what gets scored — there is no separate preview model to drift out of sync, and
no external checkout to clone. Needs mujoco + ffmpeg (and onnxruntime for --run).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import subprocess

import mujoco
import numpy as np

from env import ParkourSim, generate_course, instance_score, instance_spec

from env.sim import OBS_DIM, PLINTH_TOP, STATE_DIM, START_X, _mesh_assets, _scene_xml

OUT = pathlib.Path("renders")


def _lit_model(sim: ParkourSim) -> mujoco.MjModel:
    """The sim's own scene, re-compiled with lighting and a framebuffer big enough to render.

    The robot XML is authored for headless physics: no lights worth the name, a black backdrop,
    and a 640x480 offscreen buffer. None of that changes the dynamics, so the preview compiles a
    cosmetically patched copy rather than polluting the model the referee runs.
    """
    xml = _scene_xml(sim.frictions)
    # The schema permits exactly one <global>, and the robot already has one, so extend it.
    xml = re.sub(r"<global\b", '<global offwidth="1600" offheight="900"', xml, count=1)
    if "<visual>" not in xml:
        xml = xml.replace("<worldbody>", "<visual><global offwidth='1600' offheight='900'/>"
                          "</visual>\n  <worldbody>", 1)
    xml = xml.replace("<worldbody>",
        '<worldbody>\n'
        '    <light pos="8 -6 9" dir="-0.3 0.4 -1" directional="true" diffuse="0.6 0.6 0.57"/>\n'
        '    <light pos="34 6 9" dir="0.2 -0.4 -1" directional="true" diffuse="0.35 0.36 0.40"/>\n'
        f'    <geom type="box" pos="{sim.course.length:.2f} 0 {sim.course.final_deck + 0.9:.2f}" '
        f'size="0.04 1.2 0.02" group="2" rgba="1 .95 .2 1"/>', 1)
    return mujoco.MjModel.from_xml_string(xml, _mesh_assets())


def png(px, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{px.shape[1]}x{px.shape[0]}", "-i", "-", str(path)],
                   input=px.tobytes(), check=True)


def mp4(frame_dir, out, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frame_dir / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    print("wrote", out)


def frames_dir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    return d


def _camera():
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    return cam, opt


def render_course(sim: ParkourSim):
    m = _lit_model(sim)
    d = mujoco.MjData(m)
    d.qpos[0], d.qpos[2] = START_X, PLINTH_TOP + 0.793
    mujoco.mj_forward(m, d)
    cam, opt = _camera()

    r = mujoco.Renderer(m, height=900, width=1600)
    for name, look, dist, elev in [("01_overview_iso", sim.course.length / 2, 44, -22),
                                   ("02_side_elevation", sim.course.length / 2, 40, -6),
                                   ("03_onramp", 8.0, 14, -14),
                                   ("04_stairs_leap_drop", 20.5, 13, -14),
                                   ("05_hurdle_stepup", 27.5, 13, -14),
                                   ("06_duck_beam_slick", 38.0, 15, -14)]:
        cam.lookat[:] = [look, 0, 1.2]
        cam.distance, cam.azimuth, cam.elevation = dist, (90 if "side" in name else 132), elev
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), OUT / f"{name}.png")
    print(f"wrote 6 stills to {OUT}/")

    fd = frames_dir("_fly")
    r = mujoco.Renderer(m, height=720, width=1280)
    n = 300
    for i in range(n):
        t = i / (n - 1)
        cam.lookat[:] = [-2.0 + t * (sim.course.length + 4.0), 0, 1.0]
        cam.distance, cam.elevation = 9.0, -13 - 6 * math.sin(math.pi * t)
        cam.azimuth = 118 + 24 * math.sin(2 * math.pi * t)
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), fd / f"f{i:05d}.png")
    mp4(fd, OUT / "course_flythrough.mp4")


def film_run(sim: ParkourSim, artifact: str, seed: int, max_steps: int):
    """Drive the real sim with a real ONNX policy and film it. Scores exactly as the referee does."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(artifact, sess_options=opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]

    # Physics runs on the sim's own model; the camera renders a cosmetically lit twin that shares
    # its geometry, so qpos transfers across directly.
    view = _lit_model(sim)
    vdata = mujoco.MjData(view)
    r = mujoco.Renderer(view, height=720, width=1280)
    cam, opt = _camera()
    fd = frames_dir("_run")

    obs = sim.reset(seed)
    state = np.zeros((1, STATE_DIM), np.float32)
    every = max(1, int(round((1 / 30) / (sim.model.opt.timestep * 10))))
    fi, reason = 0, None
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        result = sim.step(np.asarray(action).ravel(), max_steps=max_steps)
        obs, reason = result.obs, result.terminal_reason
        if sim.steps % every == 0 or reason is not None:
            vdata.qpos[:] = sim.data.qpos
            mujoco.mj_forward(view, vdata)
            x, z = float(sim.data.qpos[0]), float(sim.data.qpos[2])
            cam.lookat[:] = [x + 1.2, 0, z]
            cam.distance, cam.azimuth, cam.elevation = 4.6, 128, -12
            r.update_scene(vdata, camera=cam, scene_option=opt)
            png(r.render(), fd / f"f{fi:05d}.png"); fi += 1

    score = instance_score(reason, sim.progress, sim.steps, max_steps)
    print(f"{artifact}: {reason} at {sim.max_x:.2f} m of {sim.course.length:.1f} m "
          f"({100 * sim.progress:.0f}%), score {score:.4f}")
    mp4(fd, OUT / "baseline_run.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="ONNX policy to film driving the course")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--of", type=int, default=24, help="suite size the instance is drawn from")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0,
                    help="round master seed; picks the course. Vary it to eyeball the distribution")
    ap.add_argument("--no-stills", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    level, inst_seed = instance_spec(a.instance, a.of)
    sim = ParkourSim(generate_course(a.seed), level, inst_seed)
    if not a.no_stills:
        render_course(sim)
    if a.run:
        film_run(sim, a.run, inst_seed, a.max_steps)
