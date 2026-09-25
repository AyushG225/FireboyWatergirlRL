"""Train a Discrete(36) temple policy: expert cloning, DAgger, then PPO.

Seed protocol:

- training layouts: [0, 90000); demonstrations, DAgger rollouts, and PPO
  environments all draw from this range
- validation layouts: [90000, 100000); the only seeds evaluated during training
  and the only basis for choosing a PPO checkpoint
- test layouts: >= 100000; never touched here, see scripts/evaluate_temple.py

Plain behavior cloning drifts off the expert's path within a few hundred
frames. DAgger fixes that by running the learner, labeling every visited state
with the stateless expert, and retraining on the growing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.temple_env import LOCAL_ACTION_COUNT, TempleEnv
from firewater.temple_evaluation import evaluate_temple_policy, rollout_temples
from firewater.temple_expert import TempleExpert
from firewater.train_ppo import RolloutLogger, behavior_clone

TRAIN_SEED_RANGE = (0, 90_000)
VALIDATION_SEED_START = 90_000
DAGGER_SEED_START = 1_000

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


JUMP_COMMANDS = (3, 4, 5)


def jump_weights(actions: np.ndarray, jump_weight: float) -> np.ndarray:
    """Upweight transitions where the expert tells either character to jump.

    Each character jumps on roughly 1 frame in 140, so unweighted cloning
    learns to walk into the first pool.
    """
    fire, water = np.divmod(actions, LOCAL_ACTION_COUNT)
    jumps = np.isin(fire, JUMP_COMMANDS).astype(np.float32)
    jumps += np.isin(water, JUMP_COMMANDS).astype(np.float32)
    return 1.0 + jump_weight * jumps


def dagger_beta(iteration: int, iterations: int) -> float:
    """Expert-control probability: 0.5 falling linearly to 0.

    Mixing keeps early rollouts alive long enough to reach late-episode
    states; the last two iterations run the learner alone.
    """
    return 0.5 * max(0.0, 1.0 - iteration / max(1, iterations - 2))


def make_env(
    rank: int, *, seed: int, log_dir: Path, run_name: str, observation_mode: str
):
    def _init():
        env = TempleEnv(seed_range=TRAIN_SEED_RANGE, observation_mode=observation_mode)
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
    observation_mode: str = "egocentric",
) -> tuple[np.ndarray, np.ndarray]:
    """Record the expert on each layout and verify every run succeeds."""
    experts = [TempleExpert() for _ in layout_seeds]
    observations: list[np.ndarray] = []
    actions: list[int] = []

    def choose(indices, envs, batch):
        return np.array([experts[index].action(envs[index]) for index in indices])

    def record(index, env, observation, action):
        observations.append(observation.copy())
        actions.append(action)

    episodes = rollout_temples(
        layout_seeds, choose, record=record, observation_mode=observation_mode
    )
    failed = [episode.layout_seed for episode in episodes if episode.reason != "success"]
    if failed:
        raise RuntimeError(f"temple expert failed on seeds {failed}")
    print(f"[demos] {len(layout_seeds)} layouts, {len(actions):,} transitions")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def collect_dagger(
    model: PPO,
    layout_seeds: list[int],
    *,
    beta: float,
    rng: np.random.Generator,
    observation_mode: str = "egocentric",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Roll out a beta-mixture of expert and learner; label with the expert."""
    labelers = [TempleExpert(stateless=True) for _ in layout_seeds]
    observations: list[np.ndarray] = []
    labels: list[int] = []

    def choose(indices, envs, batch):
        learner_actions, _ = model.predict(batch, deterministic=True)
        chosen = []
        for slot, index in enumerate(indices):
            expert_action = labelers[index].action(envs[index])
            observations.append(batch[slot].copy())
            labels.append(expert_action)
            use_expert = rng.random() < beta
            chosen.append(expert_action if use_expert else int(learner_actions[slot]))
        return np.array(chosen)

    episodes = rollout_temples(layout_seeds, choose, observation_mode=observation_mode)
    successes = sum(episode.reason == "success" for episode in episodes)
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        successes,
    )


class FreezeActorCallback(BaseCallback):
    """Train only the value head for the first PPO rollouts.

    After cloning, the actor is competent and the critic is untrained. Early
    advantage estimates are then noise, and full PPO updates erase the cloned
    behavior. Freezing the actor lets the critic catch up first.
    """

    def __init__(self, warmup_rollouts: int):
        super().__init__()
        self.warmup_rollouts = warmup_rollouts
        self.completed = 0

    def _actor_parameters(self):
        policy = self.model.policy
        yield from policy.mlp_extractor.policy_net.parameters()
        yield from policy.action_net.parameters()

    def _set_actor_trainable(self, trainable: bool) -> None:
        for parameter in self._actor_parameters():
            parameter.requires_grad_(trainable)

    def _on_training_start(self) -> None:
        if self.warmup_rollouts > 0:
            self._set_actor_trainable(False)
            print(
                f"[ppo] actor frozen for {self.warmup_rollouts} critic warm-up rollouts"
            )

    def _on_rollout_end(self) -> None:
        self.completed += 1
        if self.completed == self.warmup_rollouts:
            self._set_actor_trainable(True)
            print("[ppo] actor unfrozen")

    def _on_step(self) -> bool:
        return True


class ValidationCallback(BaseCallback):
    """Evaluate on validation seeds every ``every`` steps and save a checkpoint."""

    def __init__(self, seeds, every: int, output_dir: Path, run_name: str, history):
        super().__init__()
        self.seeds = seeds
        self.every = every
        self.output_dir = output_dir
        self.run_name = run_name
        self.history = history
        self.next_eval = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            name = f"{self.run_name}_ppo_{self.num_timesteps}_steps"
            self.model.save(self.output_dir / name)
            result = evaluate_temple_policy(self.model, self.seeds, controller=name)
            self.history.append(
                {"stage": "ppo", "checkpoint": name, "timesteps": self.num_timesteps}
                | result.to_dict(include_episodes=False)
            )
            print(f"[validation] {result.summary()}")
        return True


def record_validation(model, seeds, name, stage, history, output_dir):
    model.save(output_dir / name)
    result = evaluate_temple_policy(model, seeds, controller=name)
    history.append(
        {"stage": stage, "checkpoint": name, "timesteps": 0}
        | result.to_dict(include_episodes=False)
    )
    print(f"[validation] {result.summary()}")
    return result


def select_checkpoint(history: list[dict]) -> dict:
    """Highest validation success, then fewer hazards, then earlier."""
    candidates = [entry for entry in history if entry["stage"] == "ppo"]
    if not candidates:
        candidates = history
    return max(
        enumerate(candidates),
        key=lambda item: (item[1]["successes"], -item[1]["hazards"], -item[0]),
    )[1]


def run_imitation(
    model,
    args,
    *,
    demo_layouts,
    bc_epochs,
    dagger_iterations,
    dagger_layouts,
    dagger_epochs,
    validation_seeds,
    history,
    rng,
) -> dict:
    """Clone the expert, then run DAgger. Returns dataset statistics."""
    demo_seeds = list(range(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[0] + demo_layouts))
    dataset_obs, dataset_actions = generate_demonstrations(
        demo_seeds, args.observation_mode
    )
    demo_transitions = int(len(dataset_actions))
    behavior_clone(
        model,
        dataset_obs,
        dataset_actions,
        n_epochs=bc_epochs,
        batch_size=min(256, len(dataset_actions)),
        learning_rate=3e-4,
        seed=args.seed,
        sample_weights=jump_weights(dataset_actions, args.jump_weight),
    )
    record_validation(
        model, validation_seeds, f"{args.run_name}_bc", "bc", history, args.output_dir
    )

    for iteration in range(dagger_iterations):
        beta = dagger_beta(iteration, dagger_iterations)
        start = DAGGER_SEED_START + iteration * dagger_layouts
        seeds = list(range(start, start + dagger_layouts))
        new_obs, new_actions, successes = collect_dagger(
            model, seeds, beta=beta, rng=rng, observation_mode=args.observation_mode
        )
        dataset_obs = np.concatenate([dataset_obs, new_obs])
        dataset_actions = np.concatenate([dataset_actions, new_actions])
        print(
            f"[dagger] iteration {iteration + 1}/{dagger_iterations} "
            f"beta={beta:.3f} rollout_success={successes}/{len(seeds)} "
            f"dataset={len(dataset_actions):,}"
        )
        behavior_clone(
            model,
            dataset_obs,
            dataset_actions,
            n_epochs=dagger_epochs,
            batch_size=min(256, len(dataset_actions)),
            learning_rate=3e-4,
            seed=args.seed + iteration + 1,
            sample_weights=jump_weights(dataset_actions, args.jump_weight),
        )
        name = f"{args.run_name}_dagger_{iteration + 1}"
        record_validation(
            model, validation_seeds, name, "dagger", history, args.output_dir
        )
    if dagger_iterations:
        model.save(args.output_dir / f"{args.run_name}_dagger")
    return {
        "demo_layouts": demo_layouts,
        "demo_transitions": demo_transitions,
        "dagger_iterations": dagger_iterations,
        "dagger_layouts_per_iteration": dagger_layouts,
        "final_dataset_transitions": int(len(dataset_actions)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-name", default="firewater_temple")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tb_firewater"))
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--timesteps", type=int, help="PPO timesteps")
    parser.add_argument("--demo-layouts", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--dagger-iterations", type=int)
    parser.add_argument("--dagger-layouts", type=int)
    parser.add_argument("--dagger-epochs", type=int)
    parser.add_argument("--validation-layouts", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--critic-warmup-rollouts", type=int)
    parser.add_argument(
        "--init-model",
        type=Path,
        help="start PPO from this checkpoint and skip cloning and DAgger",
    )
    parser.add_argument(
        "--observation-mode",
        choices=tuple(TempleEnv.OBSERVATION_SIZES),
        default="egocentric",
    )
    parser.add_argument("--jump-weight", type=float, default=10.0)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _configured(value, default):
    return default if value is None else value


def main():
    args = parse_args()
    quick = args.quick
    n_envs = _configured(args.n_envs, 2 if quick else 8)
    n_steps = _configured(args.n_steps, 32 if quick else 1024)
    batch_size = _configured(args.batch_size, 64 if quick else 256)
    timesteps = _configured(args.timesteps, 64 if quick else 3_000_000)
    demo_layouts = _configured(args.demo_layouts, 2 if quick else 100)
    bc_epochs = _configured(args.bc_epochs, 1 if quick else 20)
    dagger_iterations = _configured(args.dagger_iterations, 1 if quick else 12)
    dagger_layouts = _configured(args.dagger_layouts, 2 if quick else 100)
    dagger_epochs = _configured(args.dagger_epochs, 1 if quick else 4)
    validation_layouts = _configured(args.validation_layouts, 2 if quick else 100)
    validate_every = _configured(args.validate_every, 64 if quick else 250_000)
    warmup = _configured(args.critic_warmup_rollouts, 1 if quick else 10)
    if n_envs < 1 or n_steps < 2:
        raise SystemExit("--n-envs must be >= 1 and --n-steps must be >= 2")
    rollout_size = n_envs * n_steps
    if not 1 <= batch_size <= rollout_size:
        raise SystemExit(
            f"--batch-size must be between 1 and rollout size ({rollout_size})"
        )
    if min(demo_layouts, validation_layouts, validate_every) < 1 or timesteps < 0:
        raise SystemExit("training and validation counts must be positive")
    if min(bc_epochs, dagger_iterations, dagger_epochs, warmup) < 0:
        raise SystemExit("epoch, iteration, and warm-up counts must be non-negative")
    if dagger_iterations and dagger_layouts < 1:
        raise SystemExit("--dagger-layouts must be positive when DAgger runs")
    last_dagger_seed = DAGGER_SEED_START + dagger_iterations * dagger_layouts
    if last_dagger_seed > VALIDATION_SEED_START:
        raise SystemExit("DAgger layouts would overlap the validation range")

    set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    for directory in (args.output_dir, args.log_dir, args.tensorboard_dir):
        directory.mkdir(parents=True, exist_ok=True)
    validation_seeds = list(
        range(VALIDATION_SEED_START, VALIDATION_SEED_START + validation_layouts)
    )
    history: list[dict] = []

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and n_envs > 1 else DummyVecEnv
    vec_env = vec_cls(
        [
            make_env(
                rank,
                seed=args.seed,
                log_dir=args.log_dir,
                run_name=args.run_name,
                observation_mode=args.observation_mode,
            )
            for rank in range(n_envs)
        ]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        target_kl=0.02,
        policy_kwargs={
            "net_arch": [512, 512],
            "activation_fn": {"relu": th.nn.ReLU, "tanh": th.nn.Tanh}[args.activation],
        },
        tensorboard_log=str(args.tensorboard_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    try:
        if args.init_model is not None:
            model.set_parameters(str(args.init_model), exact_match=True)
            print(f"[temple train] initialized policy from {args.init_model}")
            record_validation(
                model,
                validation_seeds,
                f"{args.run_name}_init",
                "init",
                history,
                args.output_dir,
            )
            imitation = {"init_model": str(args.init_model)}
        else:
            imitation = run_imitation(
                model,
                args,
                demo_layouts=demo_layouts,
                bc_epochs=bc_epochs,
                dagger_iterations=dagger_iterations,
                dagger_layouts=dagger_layouts,
                dagger_epochs=dagger_epochs,
                validation_seeds=validation_seeds,
                history=history,
                rng=rng,
            )

        if timesteps > 0:
            callbacks = CallbackList(
                [
                    FreezeActorCallback(warmup),
                    ValidationCallback(
                        validation_seeds,
                        validate_every,
                        args.output_dir,
                        args.run_name,
                        history,
                    ),
                    RolloutLogger(print_freq_episodes=50),
                ]
            )
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                tb_log_name=args.run_name,
                reset_num_timesteps=True,
                progress_bar=False,
            )
            model.save(args.output_dir / f"{args.run_name}_final")
            record_validation(
                model,
                validation_seeds,
                f"{args.run_name}_ppo_final",
                "ppo",
                history,
                args.output_dir,
            )

        selected = select_checkpoint(history)
        report = {
            "seed_protocol": {
                "train": list(TRAIN_SEED_RANGE),
                "validation_start": VALIDATION_SEED_START,
                "validation_layouts": validation_layouts,
                "test_start": 100_000,
            },
            "imitation": imitation,
            "ppo_timesteps": timesteps,
            "observation_mode": args.observation_mode,
            "jump_weight": args.jump_weight,
            "activation": args.activation,
            "selection_rule": (
                "highest validation success among "
                + (
                    "PPO checkpoints"
                    if timesteps > 0
                    else "cloning and DAgger checkpoints"
                )
                + ", then fewer hazards, then earlier"
            ),
            "selected_checkpoint": selected["checkpoint"],
            "history": history,
        }
        report_path = args.output_dir / f"{args.run_name}_validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[temple train] selected {selected['checkpoint']} on validation")
        print(f"[temple train] wrote {report_path}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
