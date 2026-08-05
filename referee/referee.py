"""humanoid_parkour gym_v1 REFEREE (the scorer sandbox, run at /app/referee.py).

Owns the physics: builds the evaluation suite, steps MuJoCo, streams observations to the player
over /act, and applies the termination + scoring gates. The player sandbox only ever sees
observation vectors.

raw_score = mean instance score over all instances (see env/scoring.py). Per-instance breakdowns
go in metadata: hidden while the round is active, revealed to miners when it completes.

The course is GENERATED FROM THE ROUND SEED (`ctx.seed` -> env.course.generate_course), so its
layout is not knowable before the round opens. Every submission in the round gets the same course,
because the platform hands every evaluation in a round the same SEED — so identical resubmissions
still score identically, but a trajectory optimised offline against last round's layout is worthless
in this one.

The seed and the course digest go in metadata for audit. That is safe because metadata is hidden
while the round is active and only revealed once it completes: the generator is public
(env/course.py), so an in-round leak of the seed would let a submission regenerate the whole layout.
"""

from __future__ import annotations

import json
import time

from dataclasses import asdict

from gym_v1 import GameResult, Referee, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import RESULT_PATH

from env import ParkourSim, generate_course, instance_score, instance_spec
from env.sim import ACT_DIM, FRAME_SKIP, OBS_DIM, PHYS_DT, STATE_DIM, InvalidAction

# Sized against the referee's 900 s timeout: ~2 s of physics per instance plus HTTP.
# The round input (CONFIG_JSON) can override.
DEFAULT_NUM_INSTANCES = 24
DEFAULT_MAX_STEPS = 4000
DEFAULT_DEADLINE_MS = 500

# Everything a broken player can throw at us, all of it the SUBMISSION's fault.
#
# The vendored gym_v1 PlayerClient only converts urllib.error.URLError and TimeoutError into
# PlayerError. A player process that dies or raises mid-request surfaces as
# http.client.RemoteDisconnected (a ConnectionResetError, so an OSError) and a player that
# answers with a non-JSON body surfaces as json.JSONDecodeError — neither is a URLError, so
# both escape the client. If they escaped play_game too, the platform would score a bad
# submission as a REFEREE failure, which is the one misattribution the contract forbids.
PLAYER_FAULTS = (PlayerError, OSError, json.JSONDecodeError)

# A conforming action is ACT_DIM floats. Anything vastly longer is rejected before it reaches
# numpy, so a submission cannot spend the referee's memory budget on our behalf.
MAX_ACTION_LEN = 1024


class ParkourReferee(Referee):
    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        start = time.monotonic()
        cfg = ctx.config or {}
        n = int(cfg.get("num_instances", DEFAULT_NUM_INSTANCES))
        max_steps = int(cfg.get("max_steps_per_episode", DEFAULT_MAX_STEPS))
        deadline_ms = int(cfg.get("deadline_ms", DEFAULT_DEADLINE_MS))
        player = players[0]

        # One course for the whole round, drawn from the platform's master seed. Built before the
        # loop so a generator failure surfaces as a referee error immediately, rather than after a
        # submission has already been part-scored.
        course = generate_course(ctx.seed)

        instances = []
        total = 0.0
        for i in range(n):
            level, seed = instance_spec(i, n)
            sim = ParkourSim(course, level, seed)
            obs = sim.reset(seed)
            # Nothing identifying the instance crosses into the player sandbox. The friction
            # level is the one thing a policy is supposed to have to FEEL rather than read, so
            # it must not leak through reset() — hence seed=0 and an empty config. The ONNX
            # wrapper discards both today, but the leak must not be one player-image edit away.
            player.reset(match_id=f"{ctx.match_id}:{i}", player_index=0, seed=0, config={})

            reason = None
            while reason is None:
                # Only the player call is inside the player-fault handler. sim.step() is OUR
                # code: if it raises, that is a referee bug and must surface as a referee
                # failure, not be laundered into a zero for the submission.
                try:
                    action = player.act(observation=obs.tolist(), deadline_ms=deadline_ms)
                except PLAYER_FAULTS:
                    reason = "player_error"  # unreachable / timed out / died / garbage response
                    break
                # A submission can otherwise OOM-kill the referee (1.5Gi) with a huge action
                # list and have the failure attributed to us. Oversized is invalid, not fatal.
                if isinstance(action, (list, tuple)) and len(action) > MAX_ACTION_LEN:
                    reason = "invalid_action"
                    break
                try:
                    result = sim.step(action, max_steps=max_steps)
                except (InvalidAction, TypeError):
                    reason = "invalid_action"  # NaN / wrong shape / non-numeric
                    break
                obs, reason = result.obs, result.terminal_reason

            score = instance_score(reason, sim.progress, sim.steps, max_steps)
            total += score
            instances.append({
                "instance": i,
                "friction_level": round(level, 4),
                "terminal_reason": reason,
                "progress": round(sim.progress, 4),
                "distance_m": round(sim.max_x, 2),
                "steps": sim.steps,
                "sim_time_s": round(sim.steps * PHYS_DT * FRAME_SKIP, 2),
                "score": round(score, 4),
            })

        completed = sum(c["terminal_reason"] == "completed" for c in instances)
        raw = total / len(instances)
        return GameResult(
            raw_scores=[raw],
            winner=0 if raw > 0 else -1,
            terminal_reason="scored",
            steps=sum(c["steps"] for c in instances),
            metadata={
                "instances": instances,
                "num_instances": len(instances),
                "num_completed": completed,
                "furthest_m": max(c["distance_m"] for c in instances),
                "eval_time_in_seconds": round(time.monotonic() - start, 1),
                # Audit trail for the round's geometry: enough to reproduce the course exactly and
                # to answer a dispute, revealed only once the round closes.
                "course_seed": int(ctx.seed),
                "course_digest": course.digest,
                "course_length_m": round(course.length, 2),
                "course_segments": [
                    {"kind": s.kind, "length": round(s.length, 3)} for s in course.segs
                ],
            },
        )

    def run(self) -> None:
        """Same as the toolkit's Referee.run(), except a player that never becomes ready is
        scored as a typed SUBMISSION failure instead of a referee failure.

        Why this override exists: gym_v1's Referee.run() calls wait_until_ready() BEFORE
        play_game(), so its PlayerError escapes at a point where no /data/result.json can be
        written — and a missing result.json is attributed to the referee. But a player that
        never reports ready is exactly what a malformed ONNX artifact looks like (see
        player/launch.py: a load failure serves is_ready() False rather than dying), which is
        the submission's fault and must come back to the miner as an explained zero.

        This is NOT papering over a referee bug: the scope is one specific PlayerError from the
        readiness wait. play_game() itself is left completely unguarded, so a genuine referee
        crash still produces no result.json and is still attributed to us.
        """
        ctx = RefereeContext.from_env()
        players = [PlayerClient(url) for url in ctx.player_urls]
        try:
            for p in players:
                p.wait_until_ready(self.readiness_timeout_s)
        except PlayerError as e:
            result = GameResult(
                raw_scores=[0.0],
                winner=-1,
                terminal_reason="submission_not_ready",
                steps=0,
                metadata={
                    "error": str(e),
                    "explanation": (
                        "The submission never became ready. Usually the ONNX artifact failed to "
                        "load or does not match the required interface: inputs obs "
                        f"[batch, {OBS_DIM}] and state_in [batch, {STATE_DIM}], outputs action "
                        f"[batch, {ACT_DIM}] and state_out [batch, {STATE_DIM}], all float32, "
                        "single file with weights embedded, <= 25 MB."
                    ),
                },
            )
        else:
            result = self.play_game(ctx, players)  # unguarded: a crash here is OUR failure

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    ParkourReferee().run()
