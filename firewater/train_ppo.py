from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from firewater.evaluation import evaluate_model, format_evaluation
from firewater.firewater_env import FireWaterEnv
from firewater.scripted_demos import generate_scripted_demo

INFO_KEYWORDS = ("success", "dead", "timeout", "reason")


@dataclass(frozen=True)
class CurriculumPhase:
    slug: str
    description: str
    level_mix: tuple[int, ...]
    timesteps: int
    entropy_coef: float


# Each tuple expresses an eight-environment weighting. assign_levels() retains
# the distribution when a different number of environments is requested.
CURRICULUM = (
    CurriculumPhase(
        "level0",
        "Build basic movement and two-door coordination",
        (0, 0, 0, 0, 0, 0, 0, 0),
        1_000_000,
        0.005,
    ),
    CurriculumPhase(
        "level1",
        "Adapt to wider hazards while retaining level 0",
        (0, 1, 1, 1, 1, 1, 1, 1),
        400_000,
        0.005,
    ),
    CurriculumPhase(
        "level2",
        "Learn raised doors and precision landings",
        (0, 1, 2, 2, 2, 2, 2, 2),
        800_000,
        0.005,
    ),
    CurriculumPhase(
        "consolidate",
        "Consolidate levels 0 through 2",
        (0, 1, 1, 2, 2, 2, 2, 2),
        400_000,
        0.001,
    ),
    CurriculumPhase(
        "all_levels",
        "Explore the complete five-level curriculum",
        (0, 1, 2, 2, 3, 3, 4, 4),
        800_000,
        0.005,
    ),
    CurriculumPhase(
        "hard_levels",
        "Refine the two hardest platform levels without forgetting earlier skills",
        (0, 1, 2, 3, 3, 3, 4, 4),
        800_000,
        0.001,
    ),
    CurriculumPhase(
        "final_consolidation",
        "Consolidate a reliable policy across every level",
        (0, 1, 2, 3, 3, 4, 4, 4),
        400_000,
        0.0,
    ),
)


def make_env(
    level_id: int = 0,
    log_prefix: str | None = None,
    env_idx: int = 0,
    log_dir: str | Path = "logs",
    seed: int | None = None,
    observation_mode: str = "enhanced",
):
    """Return a factory compatible with Stable-Baselines3 vector envs."""

    def _init():
        env = FireWaterEnv(
            render_mode=None,
            level_id=level_id,
            observation_mode=observation_mode,
        )
        if seed is not None:
            env.reset(seed=seed + env_idx)
            env.action_space.seed(seed + env_idx)

        if log_prefix is None:
            return Monitor(env, info_keywords=INFO_KEYWORDS)

        destination = Path(log_dir)
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / (
            f"{log_prefix}_level{level_id}_env{env_idx}.monitor.csv"
        )
        return Monitor(env, filename=str(filename), info_keywords=INFO_KEYWORDS)

    return _init


def build_multi_level_vec_env(
    level_ids,
    log_prefix: str,
    log_dir: str | Path = "logs",
    seed: int | None = None,
    observation_mode: str = "enhanced",
) -> DummyVecEnv:
    env_fns = [
        make_env(
            level_id=level_id,
            log_prefix=log_prefix,
            env_idx=env_idx,
            log_dir=log_dir,
            seed=seed,
            observation_mode=observation_mode,
        )
        for env_idx, level_id in enumerate(level_ids)
    ]
    return DummyVecEnv(env_fns)


def build_vec_env_for_level(
    level_id: int,
    n_envs: int,
    log_prefix: str,
    log_dir: str | Path = "logs",
    seed: int | None = None,
    observation_mode: str = "enhanced",
) -> DummyVecEnv:
    return build_multi_level_vec_env(
        [level_id] * n_envs,
        log_prefix=log_prefix,
        log_dir=log_dir,
        seed=seed,
        observation_mode=observation_mode,
    )


def assign_levels(level_mix: tuple[int, ...], n_envs: int) -> list[int]:
    """Scale an ordered curriculum mix to a vector-environment count.

    When there are enough environments, every distinct configured level gets
    at least one slot. Remaining slots follow the tuple's relative weights.
    """
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if not level_mix:
        raise ValueError("level_mix cannot be empty")

    unique_levels = list(dict.fromkeys(level_mix))
    if n_envs >= len(unique_levels):
        source_counts = np.asarray(
            [level_mix.count(level_id) for level_id in unique_levels],
            dtype=np.float64,
        )
        target_counts = n_envs * source_counts / len(level_mix)
        assigned_counts = np.ones(len(unique_levels), dtype=np.int64)

        for _ in range(n_envs - len(unique_levels)):
            # Fill the largest shortfall from the desired weighted allocation.
            index = int(np.argmax(target_counts - assigned_counts))
            assigned_counts[index] += 1

        return [
            level_id
            for level_id, count in zip(unique_levels, assigned_counts, strict=True)
            for _ in range(int(count))
        ]

    # With fewer slots than distinct levels, retain broad coverage by sampling
    # evenly across the original ordered mix.
    indices = np.floor(np.arange(n_envs) * len(level_mix) / n_envs).astype(int)
    return [level_mix[index] for index in indices]


class RolloutLogger(BaseCallback):
    """Periodically print return, length, and terminal-reason rates."""

    def __init__(self, print_freq_episodes: int = 50, verbose: int = 0):
        super().__init__(verbose)
        if print_freq_episodes < 1:
            raise ValueError("print_freq_episodes must be at least 1")
        self.print_freq_episodes = print_freq_episodes
        self._reset_buffers()

    def _reset_buffers(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_hazards = []
        self.ep_timeouts = []
        self.ep_levels = []

    def _on_training_start(self) -> None:
        # Ask first: a failed get_attr kills a SubprocVecEnv worker, which then
        # surfaces later as a BrokenPipeError on the next step.
        if self.training_env.has_attr("level_id"):
            level_ids = self.training_env.get_attr("level_id")
            print(f"[rollout] vector environment levels: {level_ids}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue

            self.ep_rewards.append(float(episode["r"]))
            self.ep_lengths.append(int(episode["l"]))
            self.ep_successes.append(bool(episode.get("success", info.get("success"))))
            self.ep_hazards.append(bool(episode.get("dead", info.get("dead"))))
            self.ep_timeouts.append(bool(episode.get("timeout", info.get("timeout"))))
            self.ep_levels.append(int(info.get("level_id", -1)))

        if len(self.ep_rewards) >= self.print_freq_episodes:
            per_level = []
            level_array = np.asarray(self.ep_levels)
            success_array = np.asarray(self.ep_successes)
            hazard_array = np.asarray(self.ep_hazards)
            timeout_array = np.asarray(self.ep_timeouts)
            for level_id in sorted(set(self.ep_levels)):
                selected = level_array == level_id
                per_level.append(
                    f"L{level_id}:n={selected.sum()} "
                    f"s={success_array[selected].mean():.0%} "
                    f"h={hazard_array[selected].mean():.0%} "
                    f"t={timeout_array[selected].mean():.0%}"
                )
            print(
                "[rollout] "
                f"episodes={len(self.ep_rewards)} "
                f"return={np.mean(self.ep_rewards):.3f} "
                f"length={np.mean(self.ep_lengths):.1f} "
                f"success={np.mean(self.ep_successes):.1%} "
                f"hazard={np.mean(self.ep_hazards):.1%} "
                f"timeout={np.mean(self.ep_timeouts):.1%} "
                f"levels=[{' | '.join(per_level)}]"
            )
            self._reset_buffers()

        return True


def load_demos(
    pattern: str = "demo_level*.npz",
    include_scripted: bool = False,
    observation_mode: str = "enhanced",
):
    """Load and validate discrete-action demonstrations."""
    paths = sorted(glob.glob(pattern))
    expected_obs_shape = FireWaterEnv(
        observation_mode=observation_mode
    ).observation_space.shape
    n_actions = FireWaterEnv().action_space.n
    observations = []
    actions = []

    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                obs = np.asarray(data["obs"])
                acts = np.asarray(data["acts"])
                saved_level = (
                    int(np.asarray(data["level_id"]).item())
                    if "level_id" in data.files
                    else None
                )

            if obs.ndim != 2:
                raise ValueError(
                    f"observations have shape {obs.shape}; expected a 2D array"
                )
            if acts.ndim != 1 or len(acts) != len(obs):
                raise ValueError("actions must be a 1D array matching observation count")
            if len(obs) == 0:
                raise ValueError("demonstration is empty")
            if not np.isfinite(obs).all():
                raise ValueError("observations contain NaN or infinity")
            if not np.issubdtype(acts.dtype, np.integer):
                raise ValueError("actions must use an integer dtype")
            if np.any((acts < 0) | (acts >= n_actions)):
                raise ValueError(f"actions must be between 0 and {n_actions - 1}")

            if obs.shape[1:] != expected_obs_shape:
                legacy_shape = (FireWaterEnv.LEGACY_OBSERVATION_SIZE,)
                if (
                    observation_mode not in ("enhanced_v1", "enhanced")
                    or obs.shape[1:] != legacy_shape
                ):
                    raise ValueError(
                        f"observations have shape {obs.shape}; expected "
                        f"(N, {expected_obs_shape[0]})"
                    )

                level_id = saved_level
                if level_id is None:
                    match = re.search(r"level(\d+)", Path(path).name)
                    level_id = int(match.group(1)) if match else None
                if level_id is None:
                    raise ValueError(
                        "legacy observations need level_id metadata or a "
                        "'level<N>' filename for enhanced replay"
                    )

                replay_env = FireWaterEnv(
                    level_id=level_id,
                    observation_mode=observation_mode,
                )
                replayed_obs = []
                replay_observation, _ = replay_env.reset()
                try:
                    for action_index, action in enumerate(acts):
                        replayed_obs.append(replay_observation)
                        replay_observation, _, terminated, truncated, _ = replay_env.step(
                            action
                        )
                        if (terminated or truncated) and action_index + 1 < len(acts):
                            raise ValueError(
                                "legacy actions continue after the replayed episode ends"
                            )
                finally:
                    replay_env.close()
                obs = np.asarray(replayed_obs, dtype=np.float32)
                print(f"[BC] Replayed legacy observations for level {level_id}: {path}")

            # Keyboard recordings often spend many frames idle before the
            # player begins. Those duplicate spawn states teach a cloned
            # policy to choose no-op at reset and never enter the useful part
            # of the demonstration, so trim recorder latency only at the head.
            non_idle = np.flatnonzero(acts != 0)
            if not len(non_idle):
                raise ValueError("demonstration contains only no-op actions")
            first_action = int(non_idle[0])
            obs = obs[first_action:]
            acts = acts[first_action:]

            observations.append(obs.astype(np.float32, copy=False))
            actions.append(acts.astype(np.int64, copy=False))
            print(f"[BC] Loaded {path}: {len(acts)} transitions")
        except (KeyError, OSError, ValueError) as exc:
            print(f"[BC] Ignoring invalid demonstration {path}: {exc}")

    if include_scripted:
        for level_id in range(5):
            demo = generate_scripted_demo(
                level_id,
                observation_mode=observation_mode,
            )
            observations.append(demo.observations)
            actions.append(demo.actions)
            print(
                f"[BC] Generated scripted level {level_id}: "
                f"{len(demo.actions)} transitions"
            )

    if not observations:
        print(f"[BC] No valid demonstrations match {pattern!r}; skipping BC.")
        return None, None

    demo_obs = np.concatenate(observations)
    demo_acts = np.concatenate(actions)
    print(f"[BC] Loaded {len(demo_acts)} total transitions")
    return demo_obs, demo_acts


def behavior_clone(
    model: PPO,
    demo_obs: np.ndarray,
    demo_acts: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    sample_weights: np.ndarray | None = None,
):
    """Warm-start an SB3 PPO policy with supervised action prediction.

    ``sample_weights`` optionally scales each transition's loss. Rare but
    decisive actions, such as the single jump frame before a pool, otherwise
    contribute almost nothing to the average.
    """
    if demo_obs is None or demo_acts is None:
        print("[BC] No demonstrations provided; skipping BC.")
        return
    if n_epochs < 1:
        print("[BC] n_epochs is zero; skipping BC.")
        return
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(demo_obs) != len(demo_acts):
        raise ValueError("demo_obs and demo_acts must have the same length")
    if sample_weights is not None and len(sample_weights) != len(demo_acts):
        raise ValueError("sample_weights must match the number of transitions")
    if demo_obs.shape[1:] != model.observation_space.shape:
        raise ValueError(
            f"demo observation shape {demo_obs.shape[1:]} does not match "
            f"model shape {model.observation_space.shape}"
        )

    policy = model.policy
    policy.train()
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    print(f"[BC] Training for {n_epochs} epochs on {len(demo_acts)} transitions")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        shuffled_indices = rng.permutation(len(demo_acts))

        for start in range(0, len(demo_acts), batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]
            obs_batch = th.as_tensor(
                demo_obs[batch_indices],
                dtype=th.float32,
                device=policy.device,
            )
            acts_batch = th.as_tensor(
                demo_acts[batch_indices],
                dtype=th.long,
                device=policy.device,
            )

            distribution = policy.get_distribution(obs_batch)
            log_prob = distribution.log_prob(acts_batch)
            if sample_weights is None:
                loss = -log_prob.mean()
            else:
                weights = th.as_tensor(
                    sample_weights[batch_indices],
                    dtype=th.float32,
                    device=policy.device,
                )
                loss = -(weights * log_prob).sum() / weights.sum()

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"[BC] epoch {epoch + 1}/{n_epochs}: loss={epoch_loss / n_batches:.4f}")


def evaluate_per_level(
    model: PPO,
    levels=(0, 1, 2, 3, 4),
    n_episodes: int = 10,
    seed: int = 0,
):
    results = evaluate_model(
        model,
        levels=levels,
        episodes=n_episodes,
        deterministic=True,
        seed=seed,
    )
    print(format_evaluation(results))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with behavior cloning and a five-level curriculum"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small end-to-end smoke training instead of the full curriculum",
    )
    parser.add_argument("--resume", type=Path, help="Resume an SB3 PPO checkpoint")
    parser.add_argument(
        "--start-phase",
        type=int,
        choices=range(1, len(CURRICULUM) + 1),
        default=1,
        help="1-based curriculum phase to start from (use with --resume for phase > 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", help="Checkpoint/log prefix")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tb_firewater"))
    parser.add_argument("--demo-pattern", default="demo_level*.npz")
    parser.add_argument("--no-bc", action="store_true")
    parser.add_argument(
        "--no-scripted-demos",
        action="store_true",
        help="Exclude the built-in successful demonstrations from behavior cloning",
    )
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="Override the curriculum entropy schedule with one coefficient",
    )
    parser.add_argument("--timesteps-scale", type=float)
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save an in-phase checkpoint every N environment steps (0 disables)",
    )
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--observation-mode",
        choices=("legacy", "enhanced_v1", "enhanced"),
        default="enhanced",
        help="Enhanced mode exposes goals, platforms, and full hazard geometry",
    )
    return parser.parse_args()


def resolve_settings(args):
    settings = {
        "run_name": args.run_name
        or ("firewater_quick" if args.quick else "firewater_ppo"),
        "n_envs": args.n_envs if args.n_envs is not None else (5 if args.quick else 8),
        "n_steps": args.n_steps
        if args.n_steps is not None
        else (16 if args.quick else 1024),
        "batch_size": args.batch_size
        if args.batch_size is not None
        else (40 if args.quick else 256),
        "bc_epochs": args.bc_epochs
        if args.bc_epochs is not None
        else (1 if args.quick else 50),
        "eval_episodes": (
            args.eval_episodes
            if args.eval_episodes is not None
            else (1 if args.quick else 5)
        ),
        "timesteps_scale": (
            args.timesteps_scale
            if args.timesteps_scale is not None
            else (0.0001 if args.quick else 1.0)
        ),
        "save_every": args.save_every
        if args.save_every is not None
        else (0 if args.quick else 100_000),
    }

    if settings["n_envs"] < 1 or settings["n_steps"] < 2:
        raise SystemExit("--n-envs must be >= 1 and --n-steps must be >= 2")
    if settings["n_envs"] < 5:
        raise SystemExit(
            "--n-envs must be at least 5 so the final curriculum phase contains "
            "every level"
        )
    rollout_size = settings["n_envs"] * settings["n_steps"]
    if not 1 <= settings["batch_size"] <= rollout_size:
        raise SystemExit(
            f"--batch-size must be between 1 and rollout size ({rollout_size})"
        )
    if settings["timesteps_scale"] <= 0:
        raise SystemExit("--timesteps-scale must be greater than zero")
    if settings["eval_episodes"] < 1:
        raise SystemExit("--eval-episodes must be at least 1")
    if args.hidden_size < 1 or args.print_every < 1:
        raise SystemExit("--hidden-size and --print-every must be at least 1")
    if args.ent_coef is not None and args.ent_coef < 0:
        raise SystemExit("--ent-coef must be non-negative")
    if args.start_phase > 1 and args.resume is None:
        raise SystemExit("--start-phase greater than 1 requires --resume")

    return settings


def save_evaluation_report(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(level_id): result.to_dict() for level_id, result in results.items()
    }
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    settings = resolve_settings(args)
    set_random_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    selected_phases = CURRICULUM[args.start_phase - 1 :]
    first_phase = selected_phases[0]
    first_levels = assign_levels(first_phase.level_mix, settings["n_envs"])
    vec_env = build_multi_level_vec_env(
        first_levels,
        log_prefix=f"{settings['run_name']}_{first_phase.slug}",
        log_dir=args.log_dir,
        seed=args.seed,
        observation_mode=args.observation_mode,
    )

    try:
        if args.resume is not None:
            print(f"[train] Resuming {args.resume} at phase {args.start_phase}")
            model = PPO.load(
                args.resume,
                env=vec_env,
                device=args.device,
                tensorboard_log=str(args.tensorboard_dir),
            )
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
                learning_rate=1e-4,
                clip_range=0.1,
                ent_coef=(
                    args.ent_coef
                    if args.ent_coef is not None
                    else first_phase.entropy_coef
                ),
                policy_kwargs={"net_arch": [args.hidden_size, args.hidden_size]},
                seed=args.seed,
                device=args.device,
            )

            if not args.no_bc:
                demo_obs, demo_acts = load_demos(
                    args.demo_pattern,
                    include_scripted=not args.no_scripted_demos,
                    observation_mode=args.observation_mode,
                )
                behavior_clone(
                    model,
                    demo_obs,
                    demo_acts,
                    n_epochs=settings["bc_epochs"],
                    batch_size=64,
                    seed=args.seed,
                )

        best_success_rate = -1.0
        rollout_size = settings["n_envs"] * settings["n_steps"]

        if not args.no_eval:
            print("\n[eval] Before PPO")
            initial_results = evaluate_per_level(
                model,
                n_episodes=settings["eval_episodes"],
                seed=args.seed,
            )
            save_evaluation_report(
                args.output_dir / f"{settings['run_name']}_initial.json",
                initial_results,
            )

        for relative_index, phase in enumerate(selected_phases):
            phase_number = args.start_phase + relative_index
            phase_entropy_coef = (
                args.ent_coef if args.ent_coef is not None else phase.entropy_coef
            )
            model.ent_coef = phase_entropy_coef

            if relative_index > 0:
                levels = assign_levels(phase.level_mix, settings["n_envs"])
                next_env = build_multi_level_vec_env(
                    levels,
                    log_prefix=f"{settings['run_name']}_{phase.slug}",
                    log_dir=args.log_dir,
                    seed=args.seed + phase_number * 1000,
                    observation_mode=args.observation_mode,
                )
                previous_env = model.get_env()
                model.set_env(next_env)
                previous_env.close()

            scaled_timesteps = max(
                int(round(phase.timesteps * settings["timesteps_scale"])),
                rollout_size,
            )
            levels = model.get_env().get_attr("level_id")
            print(
                f"\n[train] Phase {phase_number}/{len(CURRICULUM)}: {phase.description}\n"
                f"[train] levels={levels} timesteps={scaled_timesteps:,} "
                f"ent_coef={phase_entropy_coef:g}"
            )

            callbacks = [RolloutLogger(print_freq_episodes=args.print_every)]
            if settings["save_every"] > 0:
                callbacks.append(
                    CheckpointCallback(
                        save_freq=max(
                            settings["save_every"] // settings["n_envs"],
                            1,
                        ),
                        save_path=str(args.output_dir),
                        name_prefix=f"{settings['run_name']}_{phase.slug}",
                    )
                )

            model.learn(
                total_timesteps=scaled_timesteps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=False,
                tb_log_name=settings["run_name"],
            )

            phase_path = args.output_dir / (
                f"{settings['run_name']}_phase{phase_number}_{phase.slug}"
            )
            model.save(phase_path)
            print(f"[train] Saved phase checkpoint: {phase_path}.zip")

            if not args.no_eval:
                print(f"\n[eval] After phase {phase_number}")
                results = evaluate_per_level(
                    model,
                    n_episodes=settings["eval_episodes"],
                    seed=args.seed,
                )
                save_evaluation_report(phase_path.with_suffix(".json"), results)

                mean_success_rate = float(
                    np.mean([result.success_rate for result in results.values()])
                )
                if mean_success_rate > best_success_rate:
                    best_success_rate = mean_success_rate
                    best_path = args.output_dir / f"{settings['run_name']}_best"
                    model.save(best_path)
                    print(
                        f"[train] New best mean success rate: {mean_success_rate:.1%}; "
                        f"saved {best_path}.zip"
                    )

        final_path = args.output_dir / f"{settings['run_name']}_final"
        model.save(final_path)
        print(f"\n[train] Finished. Final checkpoint: {final_path}.zip")
    finally:
        active_model = locals().get("model")
        if active_model is not None and active_model.get_env() is not None:
            model.get_env().close()
        else:
            vec_env.close()


if __name__ == "__main__":
    main()
