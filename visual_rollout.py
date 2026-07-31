import argparse

from firewater_env import FireWaterEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a random-agent rollout")
    parser.add_argument("--level", type=int, choices=range(5), default=0)
    parser.add_argument("--steps", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    env = FireWaterEnv(render_mode="human", level_id=args.level)
    obs, _ = env.reset()
    print("initial obs shape:", obs.shape)

    try:
        for t in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"t={t:03d} action={action} reward={reward:+.3f} "
                f"terminated={terminated} truncated={truncated}"
            )

            if env.window_closed:
                return

            if terminated or truncated:
                print(f"Episode finished: {info['reason']}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
