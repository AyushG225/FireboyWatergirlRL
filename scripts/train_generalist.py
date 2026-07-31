from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.firewater_env import FireWaterEnv
from firewater.generalization import evaluate_generalist, format_generalization
from firewater.generalized_planner import plan_level
from firewater.procedural_levels import held_out_seeds, training_seeds
from firewater.train_ppo import RolloutLogger, behavior_clone


INFO_KEYWORDS = ("success", "dead", "timeout", "reason", "layout_seed")


def make_procedural_env(
    rank: int,
    *,
    seed: int,
    log_dir: Path,
    run_name: str,
):
    def _init():
        env = FireWaterEnv(
            procedural=True,
            observation_mode="generalized",
            procedural_seed_range=(0, 100_000),
        )
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(
            env,
            filename=str(log_dir / f"{run_name}_env{rank}.monitor.csv"),
            info_keywords=INFO_KEYWORDS,
        )

    return _init


def build_procedural_vec_env(
    n_envs: int,
    *,
    seed: int,
    log_dir: Path,
    run_name: str,
) -> DummyVecEnv:
    return DummyVecEnv(
        [
            make_procedural_env(
                rank,
                seed=seed,
                log_dir=log_dir,
                run_name=run_name,
            )
            for rank in range(n_envs)
        ]
    )


def generate_planner_demonstrations(
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []
    for index, layout_seed in enumerate(seeds, start=1):
        env = FireWaterEnv(
            procedural=True,
            level_seed=layout_seed,
            observation_mode="generalized",
        )
        observation, _ = env.reset(seed=0)
        try:
            plan = plan_level(env)
            observation, _ = env.reset(options={"level_seed": layout_seed})
            for action in plan.actions:
                observations.append(observation.copy())
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if info.get("reason") != "success":
                raise RuntimeError(
                    f"expert replay failed on layout seed {layout_seed}: {info}"
                )
        finally:
            env.close()
        if index % 25 == 0 or index == len(seeds):
            print(
                f"[expert] planned {index}/{len(seeds)} layouts; "
                f"transitions={len(actions):,}"
            )

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def save_evaluation(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one geometry-only PPO policy on randomized levels"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name", default="firewater_generalist")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tb_firewater"))
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--demo-layouts", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--eval-layouts", type=int)
    parser.add_argument("--no-bc", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_settings(args):
    def configured(value, default):
        return default if value is None else value

    settings = {
        "n_envs": configured(args.n_envs, 5 if args.quick else 8),
        "n_steps": configured(args.n_steps, 16 if args.quick else 1024),
        "batch_size": configured(args.batch_size, 40 if args.quick else 256),
        "timesteps": configured(args.timesteps, 80 if args.quick else 3_000_000),
        "demo_layouts": configured(
            args.demo_layouts,
            3 if args.quick else 300,
        ),
        "bc_epochs": (
            args.bc_epochs if args.bc_epochs is not None else (1 if args.quick else 20)
        ),
        "eval_layouts": configured(
            args.eval_layouts,
            3 if args.quick else 100,
        ),
    }
    if settings["n_envs"] < 1 or settings["n_steps"] < 2:
        raise SystemExit("--n-envs must be >= 1 and --n-steps must be >= 2")
    rollout_size = settings["n_envs"] * settings["n_steps"]
    if not 1 <= settings["batch_size"] <= rollout_size:
        raise SystemExit(
            f"--batch-size must be between 1 and rollout size ({rollout_size})"
        )
    for key in ("timesteps", "demo_layouts", "eval_layouts"):
        if settings[key] < 1:
            raise SystemExit(f"--{key.replace('_', '-')} must be at least 1")
    if settings["bc_epochs"] < 0:
        raise SystemExit("--bc-epochs must be non-negative")
    if args.hidden_size < 1:
        raise SystemExit("--hidden-size must be at least 1")
    if args.ent_coef < 0 or args.learning_rate <= 0:
        raise SystemExit("entropy must be non-negative and learning rate positive")
    return settings


def main():
    args = parse_args()
    settings = resolve_settings(args)
    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_procedural_vec_env(
        settings["n_envs"],
        seed=args.seed,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )
    try:
        if args.resume is not None:
            model = PPO.load(
                args.resume,
                env=vec_env,
                device=args.device,
                tensorboard_log=str(args.tensorboard_dir),
            )
            model.ent_coef = args.ent_coef
            print(f"[train] resumed {args.resume}")
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=str(args.tensorboard_dir),
                n_steps=settings["n_steps"],
                batch_size=settings["batch_size"],
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=args.learning_rate,
                clip_range=0.1,
                ent_coef=args.ent_coef,
                policy_kwargs={
                    "net_arch": [args.hidden_size, args.hidden_size]
                },
                seed=args.seed,
                device=args.device,
            )
            if not args.no_bc and settings["bc_epochs"] > 0:
                demo_obs, demo_actions = generate_planner_demonstrations(
                    training_seeds(settings["demo_layouts"])
                )
                behavior_clone(
                    model,
                    demo_obs,
                    demo_actions,
                    n_epochs=settings["bc_epochs"],
                    batch_size=256,
                    learning_rate=3e-4,
                    seed=args.seed,
                )

        eval_seeds = held_out_seeds(settings["eval_layouts"])
        before = evaluate_generalist(model, eval_seeds, policy_seed=args.seed)
        print(f"\n[eval] before PPO\n{format_generalization(before)}")
        save_evaluation(
            args.output_dir / f"{args.run_name}_before.json",
            before,
        )

        callbacks = [RolloutLogger(print_freq_episodes=100)]
        if args.save_every > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(args.save_every // settings["n_envs"], 1),
                    save_path=str(args.output_dir),
                    name_prefix=args.run_name,
                )
            )
        model.learn(
            total_timesteps=settings["timesteps"],
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            tb_log_name=args.run_name,
        )

        final_path = args.output_dir / f"{args.run_name}_final"
        model.save(final_path)
        after = evaluate_generalist(model, eval_seeds, policy_seed=args.seed)
        print(f"\n[eval] after PPO\n{format_generalization(after)}")
        save_evaluation(final_path.with_suffix(".json"), after)
        print(f"[train] saved {final_path}.zip")
    finally:
        active_model = locals().get("model")
        if active_model is not None and active_model.get_env() is not None:
            active_model.get_env().close()
        else:
            vec_env.close()


if __name__ == "__main__":
    main()
