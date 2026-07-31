import argparse
from pathlib import Path

from stable_baselines3 import PPO
from firewater.evaluation import observation_mode_for_model
from firewater.firewater_env import FireWaterEnv


PREFERRED_MODELS = (
    Path("checkpoints/firewater_refined_final"),
    Path("checkpoints/firewater_ppo_best"),
    Path("checkpoints/firewater_ppo_final"),
    Path("ppo_firewater"),
)


def checkpoint_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".zip").exists()


def preferred_model() -> Path | None:
    """Return the strongest conventional local checkpoint that exists."""
    return next((path for path in PREFERRED_MODELS if checkpoint_exists(path)), None)


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play")
    parser.add_argument(
        "--model",
        help="PPO checkpoint path (defaults to the strongest known local model)",
    )
    parser.add_argument("--level", type=int, choices=range(5), default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of taking deterministic actions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    model_path = Path(args.model) if args.model else preferred_model()
    if model_path is None or not checkpoint_exists(model_path):
        raise SystemExit(
            f"Model checkpoint not found: {args.model or 'no default available'}. "
            "Train one with `python train_ppo.py` or pass --model PATH."
        )

    print(f"Loading checkpoint: {model_path}")
    model = PPO.load(model_path)
    env = FireWaterEnv(
        render_mode="human",
        level_id=args.level,
        observation_mode=observation_mode_for_model(model),
    )

    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0

            for step_idx in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if env.window_closed:
                    return

                if terminated or truncated:
                    print(
                        f"episode={episode} steps={step_idx + 1} "
                        f"return={total_reward:.3f} reason={info['reason']}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
