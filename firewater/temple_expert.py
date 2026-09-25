from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from firewater.temple_env import TempleEnv, encode_joint_action


@dataclass(frozen=True)
class TempleRun:
    layout_seed: int
    actions: tuple[int, ...]
    simultaneous_frames: int
    total_reward: float
    reason: str


@dataclass
class TempleExpert:
    """Closed-loop geometry controller for the alternating-lift topology."""

    stages: dict[str, int] = field(default_factory=lambda: {"fire": 0, "water": 0})

    def reset(self) -> None:
        self.stages = {"fire": 0, "water": 0}

    def action(self, env: TempleEnv) -> int:
        fire_action = self._character_action(env, "fire")
        water_action = self._character_action(env, "water")
        return encode_joint_action(fire_action, water_action)

    def _character_action(self, env: TempleEnv, who: str) -> int:
        if getattr(env, f"{who}_at_goal"):
            return 0

        stage = self.stages[who]
        x = getattr(env, f"{who}_x")
        y = getattr(env, f"{who}_y")

        if stage >= len(env.level.moving_platforms):
            goal_x = env.level.fire_goal[0] if who == "fire" else env.level.water_goal[0]
            return self._move_toward(env, who, goal_x)

        lift = env.level.moving_platforms[stage]
        lift_rect = env.moving_rects[stage]
        lift_center = 0.5 * (lift.x1 + lift.x2)
        at_upper_landing = (
            abs(y - lift.upper_y) <= 5.0 and lift.x1 - 5.0 <= x <= lift.x2 + 5.0
        )
        if at_upper_landing:
            self.stages[who] += 1
            next_target = (
                env.level.fire_goal[0]
                if who == "fire" and self.stages[who] == 4
                else env.level.water_goal[0]
                if who == "water" and self.stages[who] == 4
                else 0.5
                * (
                    env.level.moving_platforms[self.stages[who]].x1
                    + env.level.moving_platforms[self.stages[who]].x2
                )
            )
            return 2 if next_target > x else 1

        direction = 1 if lift_center > x else -1

        # Floors 1–3 end just before their next lift. Wait at the edge until
        # the platform is level, then jump aboard.
        current_floor = lift.lower_y
        approaching_edge = (
            stage > 0
            and abs(y - current_floor) <= 7.0
            and (
                (direction < 0 and x <= lift.x2 + 42.0)
                or (direction > 0 and x >= lift.x1 - 42.0)
            )
        )
        if approaching_edge:
            if abs(lift_rect[1] - current_floor) > 9.0:
                return 0
            return 1 if direction < 0 else 2

        # Once centered on a lift, wait for it to carry the character up.
        on_lift = lift.x1 - 3.0 <= x <= lift.x2 + 3.0 and abs(y - lift_rect[1]) <= 5.0
        if on_lift and lift_rect[1] > lift.upper_y + 4.0:
            return 0

        return self._move_toward(env, who, lift_center)

    def _move_toward(self, env: TempleEnv, who: str, target_x: float) -> int:
        x = getattr(env, f"{who}_x")
        y = getattr(env, f"{who}_y")
        on_ground = getattr(env, f"{who}_on_ground")
        if abs(target_x - x) <= 9.0:
            return 0
        direction = 1 if target_x > x else -1

        if on_ground and self._deadly_hazard_ahead(
            env,
            who,
            x,
            y,
            direction,
        ):
            return 5 if direction > 0 else 4
        return 2 if direction > 0 else 1

    @staticmethod
    def _deadly_hazard_ahead(
        env: TempleEnv,
        who: str,
        x: float,
        y: float,
        direction: int,
    ) -> bool:
        for hazard in env.level.hazards:
            if hazard.kind == "lava" and who == "fire":
                continue
            if hazard.kind == "water" and who == "water":
                continue
            x1, y1, x2, _ = hazard.rect
            if abs(y - y1) > 18.0:
                continue
            # With a solid ceiling above each corridor, jumping too early
            # makes the character land in the pool. Trigger near the lip so
            # the pool is centered under the available clipped jump arc.
            if direction > 0 and 0.0 <= x1 - x <= 34.0:
                return True
            if direction < 0 and 0.0 <= x - x2 <= 34.0:
                return True
        return False


def run_temple_expert(
    env: TempleEnv,
    *,
    max_steps: int | None = None,
) -> TempleRun:
    """Run and verify the closed-loop expert in the real environment."""
    expert = TempleExpert()
    actions: list[int] = []
    total_reward = 0.0
    simultaneous_frames = 0
    info = {"reason": None}
    limit = env.max_steps if max_steps is None else min(max_steps, env.max_steps)

    for _ in range(limit):
        action = expert.action(env)
        fire_action, water_action = divmod(action, 6)
        simultaneous_frames += int(fire_action != 0 and water_action != 0)
        actions.append(action)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return TempleRun(
        layout_seed=int(env.layout_seed),
        actions=tuple(actions),
        simultaneous_frames=simultaneous_frames,
        total_reward=total_reward,
        reason=str(info.get("reason")),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify simultaneous temple solutions across generated seeds"
    )
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    successes = 0
    results = []
    for layout_seed in range(args.seed_start, args.seed_start + args.count):
        env = TempleEnv(level_seed=layout_seed)
        env.reset(seed=0)
        try:
            run = run_temple_expert(env)
        finally:
            env.close()
        successes += int(run.reason == "success")
        results.append(
            {
                "layout_seed": run.layout_seed,
                "reason": run.reason,
                "steps": len(run.actions),
                "simultaneous_frames": run.simultaneous_frames,
                "total_reward": run.total_reward,
            }
        )
        print(
            f"seed={layout_seed} reason={run.reason} "
            f"steps={len(run.actions)} simultaneous={run.simultaneous_frames}"
        )
    print(f"success={successes}/{args.count}")
    if args.output is not None:
        report = {
            "seed_start": args.seed_start,
            "layouts": args.count,
            "successes": successes,
            "success_rate": successes / args.count,
            "mean_steps": sum(item["steps"] for item in results) / args.count,
            "mean_simultaneous_frames": sum(
                item["simultaneous_frames"] for item in results
            )
            / args.count,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
