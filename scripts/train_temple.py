from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.temple_env import TempleEnv
from firewater.temple_expert import TempleExpert
from firewater.train_ppo import RolloutLogger, behavior_clone


INFO_KEYWORDS = (
    "success",
    "dead",
    "timeout",
    "reason",
    "layout_seed",
    "gems_collected",
    "buttons_active",
    "gates_open",
)


def make_env(rank: int, *, seed: int, log_dir: Path, run_name: str):
    def _init():
        env = TempleEnv(seed_range=(0, 100_000))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(
            env,
            filename=str(log_dir / f"{run_name}_env{rank}.monitor.csv"),
            info_keywords=INFO_KEYWORDS,
        )

    return _init


def generate_demonstrations(
    layout_seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for index, layout_seed in enumerate(layout_seeds, start=1):
        env = TempleEnv(level_seed=layout_seed)
        observation, _ = env.reset(seed=0)
        expert = TempleExpert()
        try:
            for _ in range(env.max_steps):
                action = expert.action(env)
                observations.append(observation.copy())
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if info.get("reason") != "success":
                raise RuntimeError(
                    f"temple expert failed on seed {layout_seed}: {info}"
                )
        finally:
            env.close()
        print(
            f"[temple expert] {index}/{len(layout_seeds)} seed={layout_seed} "
            f"transitions={len(actions):,}"
        )
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def evaluate(model: PPO, seeds: list[int]) -> tuple[int, int]:
    successes = 0
    for layout_seed in seeds:
        env = TempleEnv(level_seed=layout_seed)
        observation, _ = env.reset(seed=0)
        info = {"reason": None}
        try:
            for _ in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        successes += int(info.get("reason") == "success")
    return successes, len(seeds)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone the simultaneous expert, then train PPO on temples"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-name", default="firewater_temple")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tb_firewater"))
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--demo-layouts", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--eval-layouts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _configured(value, default):
    return default if value is None else value


def main():
    args = parse_args()
    n_envs = _configured(args.n_envs, 2 if args.quick else 8)
    n_steps = _configured(args.n_steps, 32 if args.quick else 1024)
    batch_size = _configured(args.batch_size, 64 if args.quick else 256)
    timesteps = _configured(args.timesteps, 64 if args.quick else 3_000_000)
    demo_layouts = _configured(args.demo_layouts, 2 if args.quick else 100)
    bc_epochs = _configured(args.bc_epochs, 1 if args.quick else 20)
    eval_layouts = _configured(args.eval_layouts, 2 if args.quick else 50)
    if n_envs < 1 or n_steps < 2:
        raise SystemExit("--n-envs must be >= 1 and --n-steps must be >= 2")
    rollout_size = n_envs * n_steps
    if not 1 <= batch_size <= rollout_size:
        raise SystemExit(
            f"--batch-size must be between 1 and rollout size ({rollout_size})"
        )
    if min(timesteps, demo_layouts, eval_layouts) < 1 or bc_epochs < 0:
        raise SystemExit("training counts must be positive and BC epochs non-negative")

    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    vec_env = DummyVecEnv(
        [
            make_env(
                rank,
                seed=args.seed,
                log_dir=args.log_dir,
                run_name=args.run_name,
            )
            for rank in range(n_envs)
        ]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=1e-4,
        ent_coef=0.002,
        policy_kwargs={"net_arch": [512, 512]},
        tensorboard_log=str(args.tensorboard_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    try:
        demo_obs, demo_actions = generate_demonstrations(
            list(range(args.seed, args.seed + demo_layouts))
        )
        behavior_clone(
            model,
            demo_obs,
            demo_actions,
            n_epochs=bc_epochs,
            batch_size=min(256, len(demo_actions)),
            learning_rate=3e-4,
            seed=args.seed,
        )
        held_out = list(range(100_000, 100_000 + eval_layouts))
        successes, total = evaluate(model, held_out)
        print(f"[temple eval] after BC success={successes}/{total}")

        model.learn(
            total_timesteps=timesteps,
            callback=RolloutLogger(print_freq_episodes=10),
            tb_log_name=args.run_name,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        successes, total = evaluate(model, held_out)
        print(f"[temple eval] after PPO success={successes}/{total}")
        output = args.output_dir / f"{args.run_name}_final"
        model.save(output)
        print(f"[temple train] saved {output}.zip")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
