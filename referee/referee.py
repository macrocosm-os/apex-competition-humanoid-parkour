"""humanoid_parkour gym_v1 REFEREE (the scorer sandbox, run at /app/referee.py).

Owns the physics: builds the evaluation suite, steps MuJoCo, streams observations to the player
over /act, and applies the termination + scoring gates. The player sandbox only ever sees
observation vectors.

raw_score = mean instance score over all instances (see env/scoring.py). Per-instance breakdowns
go in metadata: hidden while the round is active, revealed to miners when it completes.

Note the round SEED is deliberately unused for instance generation — the course is static and
public, and a fixed suite is what makes round-to-round variance zero. See env/sim.instance_spec.
"""

from __future__ import annotations

import json
import time

from dataclasses import asdict

from gym_v1 import GameResult, Referee, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import RESULT_PATH

from env import ParkourSim, instance_score, instance_spec
from env.sim import ACT_DIM, FRAME_SKIP, OBS_DIM, PHYS_DT, STATE_DIM, InvalidAction

# Sized against the referee's 900 s timeout: ~2 s of physics per instance plus HTTP.
# The round input (CONFIG_JSON) can override.
DEFAULT_NUM_INSTANCES = 24
DEFAULT_MAX_STEPS = 4000
DEFAULT_DEADLINE_MS = 500

# Perception ablation (non-ranking diagnostic). 4 of 24 instances is ~17% more eval time --
# worst case ~43 s on top of ~258 s, against the 900 s referee timeout. Both are round-input
# overridable so the cost can be turned down without rebuilding the image; ablation_instances: 0
# disables it entirely.
DEFAULT_ABLATION_INSTANCES = 4
# 24 m, chosen by measurement over 64 positions along the course, not by taste. The two failure
# modes for a decoy offset are being IDENTICAL to the real profile (then the check reports nothing)
# and saturating the scan at +/-SCAN_CLIP more than the real profile does (then it is as
# recognisable as a block of zeros, and a replay policy can fake a delta by falling over on cue).
#
# +12 m failed both: the decoy is flat 0.8 m ground exactly where the real course is flat 0.8 m
# ground, so max |delta| was 0.000 across the whole 0-6 m start region -- the part of the course
# every current policy actually occupies. It also ran off the far end past ~39 m, saturating the
# whole scan against the distant floor.
#
# At 24 m (with wrap-around, see env/sim._obs): min |delta| 0.150 m and median 0.950 m over those
# 64 positions, never degenerate, and mean saturation 15.5% against the real profile's own 11.9%
# -- a 3.5 point gap, so it does not stand out. 28 m gives a larger min delta (0.400) but a 7.6
# point saturation gap, which is the worse trade for a check whose value is being inconspicuous.
DEFAULT_ABLATION_OFFSET_M = 24.0

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
    def _run_instance(self, ctx, player, i, n, max_steps, deadline_ms, terrain_offset=0.0):
        """Drive one instance to termination. Returns (sim, terminal_reason).

        `terrain_offset` runs it as a perception ablation (env/sim.ParkourSim) — same physics, but
        the policy is shown the terrain from further along the course.
        """
        level, seed = instance_spec(i, n)
        sim = ParkourSim(level, seed, terrain_offset=terrain_offset)
        obs = sim.reset(seed)
        # Nothing identifying the instance crosses into the player sandbox. The friction
        # level is the one thing a policy is supposed to have to FEEL rather than read, so
        # it must not leak through reset() — hence seed=0 and an empty config. The ONNX
        # wrapper discards both today, but the leak must not be one player-image edit away.
        # The ablation tag is in match_id only so our own logs are readable; a submission that
        # keyed off it would be reading a string it is not given.
        tag = f"{ctx.match_id}:{i}{'a' if terrain_offset else ''}"
        player.reset(match_id=tag, player_index=0, seed=0, config={})

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
        return sim, reason

    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        start = time.monotonic()
        cfg = ctx.config or {}
        n = int(cfg.get("num_instances", DEFAULT_NUM_INSTANCES))
        max_steps = int(cfg.get("max_steps_per_episode", DEFAULT_MAX_STEPS))
        deadline_ms = int(cfg.get("deadline_ms", DEFAULT_DEADLINE_MS))
        player = players[0]

        n_ablate = min(int(cfg.get("ablation_instances", DEFAULT_ABLATION_INSTANCES)), n)
        offset = float(cfg.get("ablation_offset_m", DEFAULT_ABLATION_OFFSET_M))

        instances = []
        total = 0.0
        for i in range(n):
            sim, reason = self._run_instance(ctx, player, i, n, max_steps, deadline_ms)
            score = instance_score(reason, sim.progress, sim.steps, max_steps)
            total += score
            instances.append({
                "instance": i,
                "friction_level": round(sim.level, 4),
                "terminal_reason": reason,
                "progress": round(sim.progress, 4),
                "distance_m": round(sim.max_x, 2),
                "steps": sim.steps,
                "sim_time_s": round(sim.steps * PHYS_DT * FRAME_SKIP, 2),
                "score": round(score, 4),
            })

        completed = sum(c["terminal_reason"] == "completed" for c in instances)
        raw = total / len(instances)

        # PERCEPTION ABLATION, non-ranking. Re-run the first `n_ablate` instances showing the
        # policy terrain from `offset` metres further along the course, and report the paired
        # difference. A policy that reads the height scan is misled and scores worse; one replaying
        # a memorised trajectory is unaffected, so delta ~= 0 is the tell. Paired against the SAME
        # instances so the difference is clean.
        #
        # This does not touch raw_scores. It is a diagnostic for the alignment checks in HANDOFF.md,
        # and a false positive must never cost a miner their score.
        ablation = None
        if n_ablate > 0 and offset != 0.0:
            abl = []
            for i in range(n_ablate):
                sim, reason = self._run_instance(ctx, player, i, n, max_steps, deadline_ms,
                                                 terrain_offset=offset)
                abl.append({
                    "instance": i,
                    "terminal_reason": reason,
                    "progress": round(sim.progress, 4),
                    "score": round(instance_score(reason, sim.progress, sim.steps, max_steps), 4),
                })
            base_mean = sum(c["score"] for c in instances[:n_ablate]) / n_ablate
            abl_mean = sum(c["score"] for c in abl) / n_ablate
            ablation = {
                "instances": abl,
                "offset_m": offset,
                "score_normal": round(base_mean, 4),
                "score_ablated": round(abl_mean, 4),
                "delta": round(base_mean - abl_mean, 4),
                "abs_delta": round(abs(base_mean - abl_mean), 4),
                # MAGNITUDE is the signal, not sign. A policy that reads the scan is perturbed by a
                # wrong one either way: measured -0.1287 for a scan-reading policy that happened to
                # do BETTER on mismatched terrain, and exactly 0.0000 for the released baseline,
                # which slices obs to indices 0-49 (tools/make_baseline.py) and never sees the scan.
                "note": ("abs_delta near zero means the policy is not reading the height scan; "
                         "sign only says whether real terrain helped (+) or hurt (-) this policy"),
            }
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
                "perception_ablation": ablation,
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
