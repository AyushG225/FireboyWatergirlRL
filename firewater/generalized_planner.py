from __future__ import annotations

import argparse
import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firewater.firewater_env import FireWaterEnv
from firewater.procedural_levels import held_out_seeds


@dataclass(frozen=True)
class AgentState:
    x: float
    y: float
    vx: float
    vy: float
    on_ground: bool


@dataclass(frozen=True)
class PlanResult:
    layout_seed: int
    actions: tuple[int, ...]
    expanded_states: int
    total_reward: float


def _state_key(state: AgentState) -> tuple[int, int, int, int, bool]:
    """Merge physically equivalent states to keep search bounded."""
    return (
        round(state.x / 3.0),
        round(state.y / 3.0),
        round(state.vx / 12.0),
        round(state.vy / 18.0),
        state.on_ground,
    )


def _circle_in_rect(
    x: float,
    y: float,
    rect: tuple[float, float, float, float],
    radius: float,
) -> bool:
    x1, y1, x2, y2 = rect
    return (x1 - radius <= x <= x2 + radius) and (y1 - radius <= y <= y2)


def _advance(
    env: FireWaterEnv,
    state: AgentState,
    local_action: int,
    deadly_rect: tuple[float, float, float, float],
) -> AgentState | None:
    """Advance one character with the exact environment physics."""
    x, y, vx, vy = state.x, state.y, state.vx, state.vy

    if local_action == 1:
        vx = -env.move_speed
    elif local_action == 2:
        vx = env.move_speed
    elif local_action == 3 and state.on_ground:
        vy = env.jump_speed

    old_y = y
    vy += env.gravity * env.dt
    x += vx * env.dt
    y += vy * env.dt
    vx *= 0.9
    if abs(vx) < 1.0:
        vx = 0.0
    x = max(0.0, min(env.W, x))
    on_ground = False

    for px1, py1, px2, py2 in env.platforms:
        if vy >= 0 and old_y <= py1 <= y and px1 <= x <= px2:
            y = py1
            vy = 0.0
            on_ground = True
            break
        if vy < 0 and old_y >= py2 >= y and px1 <= x <= px2:
            y = py2
            vy = 0.0
            break

    if y >= env.ground_y:
        y = env.ground_y
        vy = 0.0
        on_ground = True
    if y < 0.0:
        y = 0.0
        vy = 0.0

    if _circle_in_rect(x, y, deadly_rect, env.char_radius):
        return None
    return AgentState(x=x, y=y, vx=vx, vy=vy, on_ground=on_ground)


def _at_goal(
    state: AgentState,
    goal: tuple[float, float],
) -> bool:
    return (
        state.on_ground
        and abs(state.x - goal[0]) < 25.0
        and abs(state.y - goal[1]) < 15.0
    )


def _heuristic(state: AgentState, goal: tuple[float, float]) -> float:
    horizontal_steps = abs(state.x - goal[0]) / 6.67
    vertical_steps = abs(state.y - goal[1]) / 14.0
    airborne_cost = 2.0 if not state.on_ground else 0.0
    return horizontal_steps + vertical_steps + airborne_cost


def plan_character(
    env: FireWaterEnv,
    who: str,
    *,
    max_steps: int = 180,
    max_expansions: int = 150_000,
) -> tuple[list[int], int]:
    """Use weighted A* to plan one character from raw unseen geometry."""
    if who == "fire":
        start = AgentState(
            env.fire_x,
            env.fire_y,
            env.fire_vx,
            env.fire_vy,
            env.fire_on_ground,
        )
        goal = (env.fire_goal_x, env.fire_goal_y)
        deadly_rect = env.water_rect
        action_map = (0, 1, 2, 3)
    elif who == "water":
        start = AgentState(
            env.water_x,
            env.water_y,
            env.water_vx,
            env.water_vy,
            env.water_on_ground,
        )
        goal = (env.water_goal_x, env.water_goal_y)
        deadly_rect = env.lava_rect
        action_map = (0, 4, 5, 6)
    else:
        raise ValueError("who must be 'fire' or 'water'")

    if _at_goal(start, goal):
        return [], 0

    counter = itertools.count()
    nodes: dict[int, tuple[AgentState, int | None, int | None, int]] = {}
    start_id = next(counter)
    nodes[start_id] = (start, None, None, 0)
    frontier = [(1.35 * _heuristic(start, goal), start_id)]
    best_cost = {_state_key(start): 0}
    expanded = 0

    while frontier and expanded < max_expansions:
        _, node_id = heapq.heappop(frontier)
        state, parent_id, parent_action, depth = nodes[node_id]
        if best_cost.get(_state_key(state)) != depth:
            continue
        if _at_goal(state, goal):
            local_actions = []
            cursor = node_id
            while nodes[cursor][1] is not None:
                _, previous, action, _ = nodes[cursor]
                local_actions.append(int(action))
                cursor = int(previous)
            local_actions.reverse()
            return [action_map[action] for action in local_actions], expanded
        if depth >= max_steps:
            continue

        expanded += 1
        direction_action = 2 if goal[0] > state.x else 1
        opposite_action = 1 if direction_action == 2 else 2
        candidates = (
            (direction_action, 3, 0, opposite_action)
            if state.on_ground
            else (direction_action, 0, opposite_action)
        )

        for local_action in candidates:
            next_state = _advance(env, state, local_action, deadly_rect)
            if next_state is None:
                continue
            next_depth = depth + 1
            key = _state_key(next_state)
            if next_depth >= best_cost.get(key, max_steps + 1):
                continue
            best_cost[key] = next_depth
            next_id = next(counter)
            nodes[next_id] = (next_state, node_id, local_action, next_depth)
            priority = next_depth + 1.35 * _heuristic(next_state, goal)
            heapq.heappush(frontier, (priority, next_id))

    raise RuntimeError(
        f"no {who} route found after {expanded:,} expansions "
        f"(layout_seed={env.layout_seed})"
    )


def plan_level(env: FireWaterEnv) -> PlanResult:
    """Plan both players sequentially, then verify the plan in the real env."""
    if not env.procedural:
        raise ValueError("plan_level expects a procedural environment")
    layout_seed = int(env.layout_seed)
    fire_actions, fire_expanded = plan_character(env, "fire")
    water_actions, water_expanded = plan_character(env, "water")
    actions = tuple(fire_actions + water_actions)
    if len(actions) > env.max_steps:
        raise RuntimeError(
            f"routes require {len(actions)} steps but limit is {env.max_steps}"
        )

    observation, _ = env.reset(options={"level_seed": layout_seed})
    _ = observation
    info = {"reason": None}
    total_reward = 0.0
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    if info.get("reason") != "success":
        raise RuntimeError(f"planner verification failed on layout {layout_seed}: {info}")
    return PlanResult(
        layout_seed=layout_seed,
        actions=actions,
        expanded_states=fire_expanded + water_expanded,
        total_reward=total_reward,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan and verify unseen procedural Fireboy & Watergirl levels"
    )
    parser.add_argument("--seed", type=int, help="One procedural layout seed")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--save-actions", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = (
        [args.seed]
        if args.seed is not None
        else held_out_seeds(args.count, start=args.seed_start)
    )
    all_actions = {}
    for seed in seeds:
        env = FireWaterEnv(
            procedural=True,
            level_seed=seed,
            observation_mode="generalized",
        )
        env.reset(seed=0)
        try:
            result = plan_level(env)
        finally:
            env.close()
        all_actions[str(seed)] = np.asarray(result.actions, dtype=np.int64)
        print(
            f"seed={seed} success steps={len(result.actions)} "
            f"expanded={result.expanded_states:,}"
        )

    if args.save_actions is not None:
        args.save_actions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_actions, **all_actions)
        print(f"saved {len(all_actions)} plans to {args.save_actions}")


if __name__ == "__main__":
    main()
