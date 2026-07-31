from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from temple_env import TempleEnv, encode_joint_action
from temple_expert import TempleExpert


UNSEEN_SEED_START = 100_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and play a complex simultaneous-control temple"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Control both players from the keyboard instead of using the expert",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional Stable-Baselines3 checkpoint with 250 inputs and 36 actions",
    )
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def _keyboard_local(keys, *, left, right, jump) -> int:
    direction = -1 if keys[left] and not keys[right] else 1 if keys[right] and not keys[left] else 0
    wants_jump = bool(keys[jump])
    if direction < 0:
        return 4 if wants_jump else 1
    if direction > 0:
        return 5 if wants_jump else 2
    return 3 if wants_jump else 0


def main():
    args = parse_args()
    if args.manual and args.model is not None:
        raise SystemExit("--manual and --model are mutually exclusive")

    layout_seed = (
        args.seed
        if args.seed is not None
        else UNSEEN_SEED_START
        + secrets.randbelow(2**31 - UNSEEN_SEED_START)
    )
    model = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        if model.observation_space.shape != (TempleEnv.OBSERVATION_SIZE,):
            raise SystemExit(
                f"{args.model} has observation shape "
                f"{model.observation_space.shape}; temple models require "
                f"({TempleEnv.OBSERVATION_SIZE},)"
            )
        if getattr(model.action_space, "n", None) != 36:
            raise SystemExit(
                f"{args.model} has {getattr(model.action_space, 'n', '?')} "
                "actions; temple models require 36"
            )

    env = TempleEnv(render_mode="human", level_seed=layout_seed)
    observation, _ = env.reset(seed=0)
    expert = TempleExpert()
    simultaneous_frames = 0
    print(
        f"temple seed={layout_seed}; five floors, four lifts, "
        "six gems, six switches, three gates"
    )
    if args.manual:
        print("Fire: arrow keys | Water: W/A/D | R: reset | Esc/Q: quit")
    elif model is not None:
        print(f"controller={args.model}")
    else:
        print("controller=simultaneous geometry expert")

    try:
        import pygame

        previous_reset = False
        for _ in range(env.max_steps):
            if env.window_closed:
                return
            if args.manual:
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                    return
                reset_pressed = bool(keys[pygame.K_r])
                if reset_pressed and not previous_reset:
                    observation, _ = env.reset(options={"level_seed": layout_seed})
                    expert.reset()
                previous_reset = reset_pressed
                fire_action = _keyboard_local(
                    keys,
                    left=pygame.K_LEFT,
                    right=pygame.K_RIGHT,
                    jump=pygame.K_UP,
                )
                water_action = _keyboard_local(
                    keys,
                    left=pygame.K_a,
                    right=pygame.K_d,
                    jump=pygame.K_w,
                )
                action = encode_joint_action(fire_action, water_action)
            elif model is not None:
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                action = int(action)
            else:
                action = expert.action(env)

            fire_action, water_action = divmod(int(action), 6)
            simultaneous_frames += int(fire_action != 0 and water_action != 0)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(
                    f"seed={layout_seed} reason={info['reason']} "
                    f"steps={info['steps']} simultaneous_frames="
                    f"{simultaneous_frames} gems={info['gems_collected']}/"
                    f"{info['gems_total']} switches={info['buttons_active']}/"
                    f"{info['buttons_total']}"
                )
                return
    finally:
        env.close()


if __name__ == "__main__":
    main()
