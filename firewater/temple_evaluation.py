"""Batched rollouts and metrics for temple controllers.

One code path measures the scripted expert and every trained policy, so all
temple numbers in ``reports/`` share the same definitions:

- ``both_command_frames``: frames where both characters receive a non-idle
  command. This matches the original expert benchmark.
- ``both_moving_frames``: frames where both characters receive a non-idle
  command and both change position. Pushing into a wall or idling on a lift
  does not count.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from firewater.temple_env import LOCAL_ACTION_COUNT, TempleEnv
from firewater.temple_expert import TempleExpert

# choose(indices, envs, observations) -> one joint action per listed env
ChooseActions = Callable[[list[int], list[TempleEnv], np.ndarray], np.ndarray]
# record(index, env, observation, joint_action_taken)
RecordStep = Callable[[int, TempleEnv, np.ndarray, int], None]


@dataclass(frozen=True)
class TempleEpisode:
    layout_seed: int
    reason: str
    steps: int
    total_return: float
    both_command_frames: int
    both_moving_frames: int
    gems_collected: int
    gates_open: int


@dataclass(frozen=True)
class TempleEvaluation:
    controller: str
    layouts: int
    seed_start: int
    successes: int
    hazards: int
    timeouts: int
    mean_length: float
    mean_return: float
    mean_both_command_frames: float
    mean_both_moving_frames: float
    both_moving_share: float
    episodes: tuple[TempleEpisode, ...]

    @property
    def success_rate(self) -> float:
        return self.successes / self.layouts

    def to_dict(self, *, include_episodes: bool = True) -> dict:
        result = asdict(self)
        result["success_rate"] = self.success_rate
        result["hazard_rate"] = self.hazards / self.layouts
        result["timeout_rate"] = self.timeouts / self.layouts
        if not include_episodes:
            result.pop("episodes")
        return result

    def summary(self) -> str:
        return (
            f"{self.controller}: layouts={self.layouts} "
            f"success={self.success_rate:.1%} "
            f"hazard={self.hazards / self.layouts:.1%} "
            f"timeout={self.timeouts / self.layouts:.1%} "
            f"mean_length={self.mean_length:.1f} "
            f"both_moving_share={self.both_moving_share:.1%}"
        )


def rollout_temples(
    seeds: Sequence[int],
    choose: ChooseActions,
    *,
    record: RecordStep | None = None,
    observation_mode: str = "egocentric",
) -> list[TempleEpisode]:
    """Run one episode per seed in lockstep so policies can batch inference."""
    envs = [
        TempleEnv(level_seed=int(seed), observation_mode=observation_mode)
        for seed in seeds
    ]
    observations = [env.reset(seed=0)[0] for env in envs]
    returns = [0.0] * len(envs)
    command_frames = [0] * len(envs)
    moving_frames = [0] * len(envs)
    finished: dict[int, TempleEpisode] = {}
    active = list(range(len(envs)))
    try:
        while active:
            batch = np.stack([observations[index] for index in active])
            actions = np.asarray(choose(active, envs, batch)).reshape(-1)
            still_active = []
            for slot, index in enumerate(active):
                env = envs[index]
                action = int(actions[slot])
                if record is not None:
                    record(index, env, observations[index], action)
                before = (env.fire_x, env.fire_y, env.water_x, env.water_y)
                observation, reward, terminated, truncated, info = env.step(action)
                fire_command, water_command = divmod(action, LOCAL_ACTION_COUNT)
                if fire_command and water_command:
                    command_frames[index] += 1
                    fire_moved = (env.fire_x, env.fire_y) != before[:2]
                    water_moved = (env.water_x, env.water_y) != before[2:]
                    moving_frames[index] += int(fire_moved and water_moved)
                observations[index] = observation
                returns[index] += reward
                if terminated or truncated:
                    finished[index] = TempleEpisode(
                        layout_seed=int(env.layout_seed),
                        reason=str(info["reason"]),
                        steps=int(info["steps"]),
                        total_return=float(returns[index]),
                        both_command_frames=command_frames[index],
                        both_moving_frames=moving_frames[index],
                        gems_collected=int(info["gems_collected"]),
                        gates_open=int(info["gates_open"]),
                    )
                else:
                    still_active.append(index)
            active = still_active
    finally:
        for env in envs:
            env.close()
    return [finished[index] for index in range(len(envs))]


def summarize(
    controller: str,
    seeds: Sequence[int],
    episodes: Sequence[TempleEpisode],
) -> TempleEvaluation:
    reasons = [episode.reason for episode in episodes]
    total_steps = sum(episode.steps for episode in episodes)
    return TempleEvaluation(
        controller=controller,
        layouts=len(episodes),
        seed_start=int(min(seeds)),
        successes=reasons.count("success"),
        hazards=reasons.count("hazard"),
        timeouts=reasons.count("timeout"),
        mean_length=float(np.mean([episode.steps for episode in episodes])),
        mean_return=float(np.mean([episode.total_return for episode in episodes])),
        mean_both_command_frames=float(
            np.mean([episode.both_command_frames for episode in episodes])
        ),
        mean_both_moving_frames=float(
            np.mean([episode.both_moving_frames for episode in episodes])
        ),
        both_moving_share=(
            sum(episode.both_moving_frames for episode in episodes) / total_steps
        ),
        episodes=tuple(episodes),
    )


def evaluate_temple_policy(
    model,
    seeds: Sequence[int],
    *,
    controller: str = "policy",
    deterministic: bool = True,
) -> TempleEvaluation:
    seeds = [int(seed) for seed in seeds]
    mode = TempleEnv.mode_for_size(model.observation_space.shape[0])

    def choose(indices, envs, observations):
        actions, _ = model.predict(observations, deterministic=deterministic)
        return actions

    episodes = rollout_temples(seeds, choose, observation_mode=mode)
    return summarize(controller, seeds, episodes)


def evaluate_temple_expert(seeds: Sequence[int]) -> TempleEvaluation:
    seeds = [int(seed) for seed in seeds]
    experts = [TempleExpert() for _ in seeds]

    def choose(indices, envs, observations):
        return np.array([experts[index].action(envs[index]) for index in indices])

    return summarize("scripted expert", seeds, rollout_temples(seeds, choose))
