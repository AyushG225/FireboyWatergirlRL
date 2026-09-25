from __future__ import annotations

import math
from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from firewater.temple_levels import (
    ButtonSpec,
    GateSpec,
    GemSpec,
    HazardSpec,
    MovingPlatformSpec,
    TempleLevelSpec,
    generate_temple_level,
)

try:
    import pygame
except ImportError:
    pygame = None


LOCAL_ACTIONS = (
    "idle",
    "left",
    "right",
    "jump",
    "left_jump",
    "right_jump",
)
LOCAL_ACTION_COUNT = len(LOCAL_ACTIONS)


def encode_joint_action(fire_action: int, water_action: int) -> int:
    """Pack two independent local actions into one Discrete(36) action."""
    fire_action = int(fire_action)
    water_action = int(water_action)
    if not 0 <= fire_action < LOCAL_ACTION_COUNT:
        raise ValueError("fire_action must be between 0 and 5")
    if not 0 <= water_action < LOCAL_ACTION_COUNT:
        raise ValueError("water_action must be between 0 and 5")
    return fire_action * LOCAL_ACTION_COUNT + water_action


def decode_joint_action(action: int) -> tuple[int, int]:
    action = int(action)
    if not 0 <= action < LOCAL_ACTION_COUNT**2:
        raise ValueError("joint action must be between 0 and 35")
    return divmod(action, LOCAL_ACTION_COUNT)


class TempleEnv(gym.Env):
    """A multi-floor cooperative platformer with simultaneous control.

    Every RL step contains one local command for Fire and one for Water. The
    world includes elemental and neutral pools, collectible gems, character-
    specific buttons, cooperative gates, four moving lifts, and locked doors.
    Geometry is generated from a seed without exposing that seed to the policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    W = 960
    H = 720
    MAX_PLATFORMS = 12
    MAX_MOVING_PLATFORMS = 4
    MAX_HAZARDS = 12
    MAX_GEMS = 12
    MAX_BUTTONS = 8
    MAX_GATES = 4
    OBSERVATION_SIZE = 250

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        level_seed: int | None = None,
        seed_range: tuple[int, int] = (0, 100_000),
    ):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected None, "
                "'human', or 'rgb_array'."
            )
        seed_low, seed_high = map(int, seed_range)
        if seed_low < 0 or seed_high <= seed_low:
            raise ValueError("seed_range must be a non-negative [low, high) range")

        self.render_mode = render_mode
        self.level_seed = int(level_seed) if level_seed is not None else None
        self.seed_range = (seed_low, seed_high)
        self.layout_seed: int | None = None
        self.level: TempleLevelSpec | None = None

        self.gravity = 1100.0
        self.move_speed = 220.0
        self.jump_speed = -485.0
        self.dt = 1.0 / self.metadata["render_fps"]
        self.char_radius = 14.0
        self.ground_y = 660.0
        self.max_steps = 1800

        self.action_space = spaces.Discrete(LOCAL_ACTION_COUNT**2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        self.window_closed = False
        self._font_cache: dict[int, object] = {}

        self.fire_x = self.fire_y = self.fire_vx = self.fire_vy = 0.0
        self.water_x = self.water_y = self.water_vx = self.water_vy = 0.0
        self.fire_on_ground = self.water_on_ground = True
        self.fire_at_goal = self.water_at_goal = False
        self.steps = 0
        self._episode_done = False
        self.moving_rects: list[tuple[float, float, float, float]] = []
        self.button_active: dict[str, bool] = {}
        self.collected_gems: set[int] = set()
        self.just_collected: list[int] = []
        self.just_activated: list[str] = []
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        requested_seed = (
            options.get("level_seed")
            if options is not None and "level_seed" in options
            else self.level_seed
        )
        if requested_seed is None:
            requested_seed = int(
                self.np_random.integers(self.seed_range[0], self.seed_range[1])
            )
        self.layout_seed = int(requested_seed)
        self.level = generate_temple_level(
            self.layout_seed,
            width=self.W,
            height=self.H,
        )
        self.ground_y = self.level.floor_y[0]

        self.fire_x, self.fire_y = self.level.fire_start
        self.water_x, self.water_y = self.level.water_start
        self.fire_vx = self.fire_vy = 0.0
        self.water_vx = self.water_vy = 0.0
        self.fire_on_ground = self.water_on_ground = True
        self.fire_at_goal = self.water_at_goal = False
        self.steps = 0
        self._episode_done = False
        self.button_active = {button.button_id: False for button in self.level.buttons}
        self.collected_gems = set()
        self.just_collected = []
        self.just_activated = []
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False
        self.moving_rects = [
            self._moving_rect(spec, time_seconds=0.0)
            for spec in self.level.moving_platforms
        ]

        if self.render_mode == "human":
            self._ensure_render()
        return self._get_obs(), self._info(reason=None)

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("step() called after episode end; call reset() first")
        fire_action, water_action = decode_joint_action(action)
        potential_before = self._progress_potential()

        self.just_collected = []
        self.just_activated = []
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False

        self._advance_moving_platforms()
        self._apply_local_action("fire", fire_action)
        self._apply_local_action("water", water_action)
        self._physics_step()
        self.steps += 1

        self._update_buttons()
        self._collect_gems()
        self._update_goal_states()

        dead = self._fell_into_deadly_pool()
        success = self.fire_at_goal and self.water_at_goal
        timeout = self.steps >= self.max_steps
        terminated = bool(dead or success)
        truncated = bool(timeout and not terminated)

        reward = -0.002
        reward += 2.5 * (self._progress_potential() - potential_before)
        reward += 2.0 * len(self.just_collected)
        reward += 4.0 * len(self.just_activated)
        reward += 15.0 * (
            int(self.just_reached_fire_goal) + int(self.just_reached_water_goal)
        )

        reason = None
        if success:
            reward += 100.0
            reason = "success"
        elif dead:
            reward -= 30.0
            reason = "hazard"
        elif timeout:
            reward -= 10.0
            reason = "timeout"

        self._episode_done = terminated or truncated
        info = self._info(reason=reason)
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Simultaneous control and mechanics
    # ------------------------------------------------------------------
    def _apply_local_action(self, who: str, local_action: int) -> None:
        move = -1 if local_action in (1, 4) else 1 if local_action in (2, 5) else 0
        wants_jump = local_action in (3, 4, 5)
        at_goal = self.fire_at_goal if who == "fire" else self.water_at_goal
        if at_goal:
            setattr(self, f"{who}_vx", 0.0)
            setattr(self, f"{who}_vy", 0.0)
            return

        setattr(self, f"{who}_vx", move * self.move_speed)
        if wants_jump and getattr(self, f"{who}_on_ground"):
            setattr(self, f"{who}_vy", self.jump_speed)
            setattr(self, f"{who}_on_ground", False)

    def _moving_rect(
        self,
        spec: MovingPlatformSpec,
        *,
        time_seconds: float,
    ) -> tuple[float, float, float, float]:
        distance = spec.lower_y - spec.upper_y
        travel_time = distance / spec.speed
        dwell_time = 0.8
        cycle_time = 2.0 * travel_time + 2.0 * dwell_time
        cycle_position = (time_seconds + spec.phase * cycle_time) % cycle_time
        if cycle_position < dwell_time:
            top = spec.lower_y
        elif cycle_position < dwell_time + travel_time:
            top = spec.lower_y - (cycle_position - dwell_time) * spec.speed
        elif cycle_position < 2.0 * dwell_time + travel_time:
            top = spec.upper_y
        else:
            top = (
                spec.upper_y
                + (cycle_position - 2.0 * dwell_time - travel_time) * spec.speed
            )
        return (spec.x1, top, spec.x2, top + 16.0)

    def _advance_moving_platforms(self) -> None:
        previous = self.moving_rects
        next_time = (self.steps + 1) * self.dt
        updated = [
            self._moving_rect(spec, time_seconds=next_time)
            for spec in self.level.moving_platforms
        ]
        for who in ("fire", "water"):
            x = getattr(self, f"{who}_x")
            y = getattr(self, f"{who}_y")
            vy = getattr(self, f"{who}_vy")
            if vy < -1.0:
                continue
            for old_rect, new_rect in zip(previous, updated, strict=True):
                x1, old_top, x2, _ = old_rect
                if x1 - 2.0 <= x <= x2 + 2.0 and abs(y - old_top) <= 4.0:
                    setattr(self, f"{who}_y", y + new_rect[1] - old_top)
                    setattr(self, f"{who}_on_ground", True)
                    break
        self.moving_rects = updated

    def _physics_step(self) -> None:
        for who in ("fire", "water"):
            old_x = getattr(self, f"{who}_x")
            old_y = getattr(self, f"{who}_y")
            vx = getattr(self, f"{who}_vx")
            vy = getattr(self, f"{who}_vy") + self.gravity * self.dt
            x = old_x + vx * self.dt
            y = old_y + vy * self.dt
            x, y, vx, vy, on_ground = self._resolve_character(
                x,
                y,
                vx,
                vy,
                old_x,
                old_y,
            )
            setattr(self, f"{who}_x", x)
            setattr(self, f"{who}_y", y)
            setattr(self, f"{who}_vx", vx)
            setattr(self, f"{who}_vy", vy)
            setattr(self, f"{who}_on_ground", on_ground)

    def _resolve_character(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        old_x: float,
        old_y: float,
    ) -> tuple[float, float, float, float, bool]:
        radius = self.char_radius
        x = max(radius, min(self.W - radius, x))

        for gate in self.level.gates:
            if self._gate_is_open(gate):
                continue
            gx1, gy1, gx2, gy2 = gate.rect
            body_top = min(old_y, y) - 2.0 * radius
            body_bottom = max(old_y, y)
            if body_bottom < gy1 or body_top > gy2:
                continue
            if old_x + radius <= gx1 and x + radius > gx1:
                x = gx1 - radius
                vx = 0.0
            elif old_x - radius >= gx2 and x - radius < gx2:
                x = gx2 + radius
                vx = 0.0

        on_ground = False
        supports: Iterable[tuple[float, float, float, float]] = (
            *self.level.platforms,
            *self.moving_rects,
        )
        for px1, py1, px2, py2 in supports:
            if vy >= 0.0 and old_y <= py1 + 4.0 and y >= py1 and px1 <= x <= px2:
                y = py1
                vy = 0.0
                on_ground = True
                break
            old_head = old_y - 2.0 * radius
            new_head = y - 2.0 * radius
            if vy < 0.0 and old_head >= py2 >= new_head and px1 <= x <= px2:
                # Character positions track their feet, so ceiling collision
                # must use the top of the body rather than the feet. Snap the
                # head below the obstacle instead of letting the sprite phase
                # through until its feet reach the platform underside.
                y = py2 + 2.0 * radius
                vy = 0.0
                break

        if y >= self.ground_y:
            y = self.ground_y
            vy = 0.0
            on_ground = True
        if y < 2.0 * radius:
            y = 2.0 * radius
            vy = 0.0
        return x, y, vx, vy, on_ground

    def _update_buttons(self) -> None:
        for button in self.level.buttons:
            if self.button_active[button.button_id]:
                continue
            owners = (button.owner,) if button.owner != "any" else ("fire", "water")
            for who in owners:
                x = getattr(self, f"{who}_x")
                y = getattr(self, f"{who}_y")
                on_ground = getattr(self, f"{who}_on_ground")
                if (
                    on_ground
                    and abs(x - button.position[0]) <= 27.0
                    and abs(y - button.position[1]) <= 10.0
                ):
                    self.button_active[button.button_id] = True
                    self.just_activated.append(button.button_id)
                    break

    def _collect_gems(self) -> None:
        for index, gem in enumerate(self.level.gems):
            if index in self.collected_gems:
                continue
            x = getattr(self, f"{gem.owner}_x")
            y = getattr(self, f"{gem.owner}_y") - self.char_radius
            # The characters can collect while jumping, as in the original
            # game. A generous halo avoids requiring a pixel-perfect arc.
            if math.hypot(x - gem.position[0], y - gem.position[1]) <= 64.0:
                self.collected_gems.add(index)
                self.just_collected.append(index)

    def _owner_gems_complete(self, owner: str) -> bool:
        required = [
            index for index, gem in enumerate(self.level.gems) if gem.owner == owner
        ]
        return all(index in self.collected_gems for index in required)

    def _gate_is_open(self, gate: GateSpec) -> bool:
        return all(self.button_active[button] for button in gate.required_buttons)

    def _all_gates_open(self) -> bool:
        return all(self._gate_is_open(gate) for gate in self.level.gates)

    def _update_goal_states(self) -> None:
        doors_unlocked = self._all_gates_open()
        for who in ("fire", "water"):
            if getattr(self, f"{who}_at_goal"):
                continue
            goal_x, goal_y = (
                self.level.fire_goal if who == "fire" else self.level.water_goal
            )
            if (
                doors_unlocked
                and self._owner_gems_complete(who)
                and getattr(self, f"{who}_on_ground")
                and abs(getattr(self, f"{who}_x") - goal_x) < 25.0
                and abs(getattr(self, f"{who}_y") - goal_y) < 15.0
            ):
                setattr(self, f"{who}_at_goal", True)
                setattr(self, f"just_reached_{who}_goal", True)
                setattr(self, f"{who}_x", goal_x)
                setattr(self, f"{who}_y", goal_y)
                setattr(self, f"{who}_vx", 0.0)
                setattr(self, f"{who}_vy", 0.0)

    def _fell_into_deadly_pool(self) -> bool:
        radius = self.char_radius
        for hazard in self.level.hazards:
            x1, y1, x2, y2 = hazard.rect
            for who in ("fire", "water"):
                if (hazard.kind == "lava" and who == "fire") or (
                    hazard.kind == "water" and who == "water"
                ):
                    continue
                x = getattr(self, f"{who}_x")
                y = getattr(self, f"{who}_y")
                if x1 - radius <= x <= x2 + radius and y1 - radius <= y <= y2:
                    return True
        return False

    def _progress_potential(self) -> float:
        ascent = (2.0 * self.ground_y - self.fire_y - self.water_y) / (2.0 * self.H)
        interaction = 0.04 * sum(self.button_active.values())
        interaction += 0.025 * len(self.collected_gems)
        return float(ascent + interaction)

    def _info(self, *, reason: str | None) -> dict:
        return {
            "layout_seed": self.layout_seed,
            "steps": self.steps,
            "success": bool(self.fire_at_goal and self.water_at_goal),
            "dead": bool(reason == "hazard"),
            "timeout": bool(reason == "timeout"),
            "reason": reason,
            "fire_at_goal": bool(self.fire_at_goal),
            "water_at_goal": bool(self.water_at_goal),
            "gems_collected": len(self.collected_gems),
            "gems_total": len(self.level.gems),
            "buttons_active": sum(self.button_active.values()),
            "buttons_total": len(self.button_active),
            "gates_open": sum(self._gate_is_open(gate) for gate in self.level.gates),
            "gates_total": len(self.level.gates),
        }

    # ------------------------------------------------------------------
    # Geometry-only observation
    # ------------------------------------------------------------------
    def _norm_x(self, value: float) -> float:
        return float(np.clip(value / self.W * 2.0 - 1.0, -1.0, 1.0))

    def _norm_y(self, value: float) -> float:
        return float(np.clip(value / self.H * 2.0 - 1.0, -1.0, 1.0))

    def _rect_values(self, rect) -> list[float]:
        x1, y1, x2, y2 = rect
        return [
            self._norm_x(x1),
            self._norm_y(y1),
            self._norm_x(x2),
            self._norm_y(y2),
        ]

    def _get_obs(self) -> np.ndarray:
        values = [
            self._norm_x(self.fire_x),
            self._norm_y(self.fire_y),
            float(np.clip(self.fire_vx / self.move_speed, -1.0, 1.0)),
            float(np.clip(self.fire_vy / abs(self.jump_speed), -1.0, 1.0)),
            float(self.fire_on_ground),
            self._norm_x(self.water_x),
            self._norm_y(self.water_y),
            float(np.clip(self.water_vx / self.move_speed, -1.0, 1.0)),
            float(np.clip(self.water_vy / abs(self.jump_speed), -1.0, 1.0)),
            float(self.water_on_ground),
            self._norm_x(self.level.fire_goal[0]),
            self._norm_y(self.level.fire_goal[1]),
            self._norm_x(self.level.water_goal[0]),
            self._norm_y(self.level.water_goal[1]),
            float(self.fire_at_goal),
            float(self.water_at_goal),
            float(np.clip(1.0 - self.steps / self.max_steps, 0.0, 1.0)),
            self._gem_fraction("fire"),
            self._gem_fraction("water"),
            sum(self.button_active.values()) / max(1, len(self.button_active)),
            sum(self._gate_is_open(gate) for gate in self.level.gates)
            / max(1, len(self.level.gates)),
            float(self._all_gates_open()),
        ]

        for index in range(self.MAX_PLATFORMS):
            values.extend(
                self._rect_values(self.level.platforms[index])
                if index < len(self.level.platforms)
                else [-1.0] * 4
            )
        for index in range(self.MAX_MOVING_PLATFORMS):
            if index < len(self.level.moving_platforms):
                spec = self.level.moving_platforms[index]
                values.extend(
                    [
                        self._norm_x(spec.x1),
                        self._norm_x(spec.x2),
                        self._norm_y(self.moving_rects[index][1]),
                        self._norm_y(spec.upper_y),
                        self._norm_y(spec.lower_y),
                    ]
                )
            else:
                values.extend([-1.0] * 5)
        hazard_kind = {"lava": -1.0, "acid": 0.0, "water": 1.0}
        for index in range(self.MAX_HAZARDS):
            if index < len(self.level.hazards):
                hazard = self.level.hazards[index]
                values.extend(self._rect_values(hazard.rect))
                values.append(hazard_kind[hazard.kind])
            else:
                values.extend([-1.0] * 5)
        for index in range(self.MAX_GEMS):
            if index < len(self.level.gems):
                gem = self.level.gems[index]
                values.extend(
                    [
                        self._norm_x(gem.position[0]),
                        self._norm_y(gem.position[1]),
                        -1.0 if gem.owner == "fire" else 1.0,
                        float(index in self.collected_gems),
                    ]
                )
            else:
                values.extend([-1.0] * 4)
        for index in range(self.MAX_BUTTONS):
            if index < len(self.level.buttons):
                button = self.level.buttons[index]
                values.extend(
                    [
                        self._norm_x(button.position[0]),
                        self._norm_y(button.position[1]),
                        -1.0 if button.owner == "fire" else 1.0,
                        float(self.button_active[button.button_id]),
                    ]
                )
            else:
                values.extend([-1.0] * 4)
        for index in range(self.MAX_GATES):
            if index < len(self.level.gates):
                gate = self.level.gates[index]
                values.extend(self._rect_values(gate.rect))
                values.append(float(self._gate_is_open(gate)))
            else:
                values.extend([-1.0] * 5)

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.OBSERVATION_SIZE,):
            raise RuntimeError(
                f"internal observation size {observation.size}; "
                f"expected {self.OBSERVATION_SIZE}"
            )
        return observation

    def _gem_fraction(self, owner: str) -> float:
        indices = [
            index for index, gem in enumerate(self.level.gems) if gem.owner == owner
        ]
        return sum(index in self.collected_gems for index in indices) / max(
            1, len(indices)
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame(return_array=True)
        return None

    def _ensure_render(self) -> bool:
        if pygame is None:
            raise RuntimeError("pygame must be installed to render TempleEnv")
        if self.window_closed:
            return False
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Fire & Water — Procedural Temple")
                self.screen = pygame.display.set_mode((self.W, self.H))
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((self.W, self.H))
        return True

    def _font(self, size: int):
        if size not in self._font_cache:
            self._font_cache[size] = pygame.font.Font(None, size)
        return self._font_cache[size]

    def _render_frame(self, return_array=False):
        if not self._ensure_render():
            return None
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Mossy, layered temple background.
        self.screen.fill((9, 17, 16))
        for y in range(0, self.H, 24):
            shade = 22 + int(8 * math.sin(y * 0.041 + self.layout_seed % 19))
            pygame.draw.rect(self.screen, (shade, shade + 8, 18), (0, y, self.W, 24))
        for row, y in enumerate(range(22, self.H, 34)):
            offset = 24 if row % 2 else 0
            for x in range(-offset, self.W, 72):
                pygame.draw.rect(
                    self.screen,
                    (31, 43, 29),
                    (x, y, 69, 31),
                    1,
                    border_radius=3,
                )

        # Soft pools of light add depth while remaining deterministic.
        glow = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for x, y in ((480, 120), (260, 360), (720, 560)):
            pygame.draw.circle(glow, (190, 220, 150, 12), (x, y), 80)
            pygame.draw.circle(glow, (225, 240, 180, 16), (x, y), 25)
        self.screen.blit(glow, (0, 0))

        pygame.draw.rect(
            self.screen,
            (55, 60, 40),
            (0, self.ground_y, self.W, self.H - self.ground_y),
        )
        self._draw_stone_rect((0, self.ground_y, self.W, self.H), heavy=True)
        for platform in self.level.platforms:
            self._draw_stone_rect(platform)

        for index, rect in enumerate(self.moving_rects):
            self._draw_lift(rect, index)
        for hazard in self.level.hazards:
            self._draw_hazard(hazard)
        for gate in self.level.gates:
            self._draw_gate(gate)
        for button in self.level.buttons:
            self._draw_button(button)
        for index, gem in enumerate(self.level.gems):
            if index not in self.collected_gems:
                self._draw_gem(gem)

        self._draw_door("water", self.level.water_goal, self.water_at_goal)
        self._draw_door("fire", self.level.fire_goal, self.fire_at_goal)
        if not self.fire_at_goal:
            self._draw_character("fire", self.fire_x, self.fire_y)
        if not self.water_at_goal:
            self._draw_character("water", self.water_x, self.water_y)
        self._draw_vines()
        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
        if return_array:
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
        return None

    def _draw_stone_rect(self, rect, *, heavy=False) -> None:
        x1, y1, x2, y2 = map(int, rect)
        base = (77, 82, 53) if not heavy else (65, 69, 45)
        pygame.draw.rect(
            self.screen, (26, 31, 21), (x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6)
        )
        pygame.draw.rect(self.screen, base, (x1, y1, x2 - x1, y2 - y1))
        brick_h = 14
        for row, y in enumerate(range(y1, y2, brick_h)):
            offset = 18 if row % 2 else 0
            pygame.draw.line(self.screen, (43, 49, 33), (x1, y), (x2, y), 1)
            for x in range(x1 - offset, x2, 36):
                pygame.draw.line(
                    self.screen,
                    (42, 48, 32),
                    (x, y),
                    (x, min(y + brick_h, y2)),
                    1,
                )
        pygame.draw.line(self.screen, (126, 128, 76), (x1, y1), (x2, y1), 2)
        for x in range(x1 + 10, x2, 47):
            if ((x + int(self.layout_seed)) // 47) % 3 == 0:
                pygame.draw.line(
                    self.screen,
                    (21, 94, 39),
                    (x, y1),
                    (x + 5, min(y1 + 13, y2)),
                    3,
                )

    def _draw_lift(self, rect, index: int) -> None:
        x1, y1, x2, y2 = map(int, rect)
        pygame.draw.line(
            self.screen,
            (111, 99, 55),
            ((x1 + x2) // 2, int(self.level.moving_platforms[index].upper_y - 45)),
            ((x1 + x2) // 2, y1),
            2,
        )
        pygame.draw.rect(
            self.screen,
            (42, 28, 18),
            (x1 - 3, y1 - 2, x2 - x1 + 6, y2 - y1 + 4),
            border_radius=5,
        )
        pygame.draw.rect(
            self.screen, (194, 139, 32), (x1, y1, x2 - x1, y2 - y1), border_radius=5
        )
        pygame.draw.rect(
            self.screen,
            (108, 40, 137),
            (x1 + 8, y1 + 4, x2 - x1 - 16, max(4, y2 - y1 - 8)),
            border_radius=3,
        )

    def _draw_hazard(self, hazard: HazardSpec) -> None:
        colors = {
            "lava": ((255, 89, 18), (255, 194, 35)),
            "water": ((20, 116, 230), (81, 207, 255)),
            "acid": ((35, 181, 72), (123, 255, 94)),
        }
        body, highlight = colors[hazard.kind]
        x1, y1, x2, y2 = map(int, hazard.rect)
        pygame.draw.rect(
            self.screen,
            (14, 18, 16),
            (x1 - 3, y1 - 1, x2 - x1 + 6, y2 - y1 + 4),
            border_radius=7,
        )
        pygame.draw.rect(self.screen, body, (x1, y1, x2 - x1, y2 - y1), border_radius=6)
        wave = int(3 * math.sin(self.steps * 0.22 + x1 * 0.03))
        pygame.draw.line(
            self.screen, highlight, (x1 + 5, y1 + 3 + wave), (x2 - 5, y1 + 3 - wave), 3
        )

    def _draw_gate(self, gate: GateSpec) -> None:
        if self._gate_is_open(gate):
            return
        x1, y1, x2, y2 = map(int, gate.rect)
        pygame.draw.rect(self.screen, (40, 26, 17), (x1 - 4, y1, x2 - x1 + 8, y2 - y1))
        for x in range(x1, x2 + 1, 7):
            pygame.draw.line(self.screen, (176, 137, 59), (x, y1), (x, y2), 3)
        pygame.draw.circle(self.screen, (230, 186, 57), ((x1 + x2) // 2, y1 + 12), 6)

    def _draw_button(self, button: ButtonSpec) -> None:
        x, y = map(int, button.position)
        active = self.button_active[button.button_id]
        color = (255, 78, 36) if button.owner == "fire" else (57, 187, 255)
        pygame.draw.polygon(
            self.screen,
            (81, 57, 31),
            [(x - 18, y), (x - 13, y - 9), (x + 13, y - 9), (x + 18, y)],
        )
        pygame.draw.rect(
            self.screen,
            (70, 210, 88) if active else color,
            (x - 11, y - (4 if active else 10), 22, 6),
            border_radius=3,
        )

    def _draw_gem(self, gem: GemSpec) -> None:
        x, y = map(int, gem.position)
        color = (255, 42, 30) if gem.owner == "fire" else (41, 193, 255)
        halo = pygame.Surface((52, 52), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*color, 38), (26, 26), 23)
        self.screen.blit(halo, (x - 26, y - 26))
        points = [
            (x, y - 14),
            (x + 12, y - 5),
            (x + 7, y + 12),
            (x, y + 19),
            (x - 7, y + 12),
            (x - 12, y - 5),
        ]
        pygame.draw.polygon(self.screen, (18, 22, 20), points)
        inner = [
            (x, y - 11),
            (x + 9, y - 4),
            (x + 5, y + 9),
            (x, y + 15),
            (x - 5, y + 9),
            (x - 9, y - 4),
        ]
        pygame.draw.polygon(self.screen, color, inner)
        pygame.draw.line(self.screen, (255, 245, 220), (x - 4, y - 7), (x + 3, y - 5), 2)

    def _draw_door(self, owner: str, goal, reached: bool) -> None:
        x, y = map(int, goal)
        color = (255, 69, 37) if owner == "fire" else (54, 190, 255)
        unlocked = self._all_gates_open() and self._owner_gems_complete(owner)
        pygame.draw.rect(
            self.screen, (28, 24, 18), (x - 25, y - 66, 50, 66), border_radius=8
        )
        pygame.draw.rect(
            self.screen, (101, 88, 55), (x - 21, y - 62, 42, 62), border_radius=7
        )
        pygame.draw.rect(
            self.screen,
            color if unlocked else (72, 75, 68),
            (x - 15, y - 52, 30, 52),
            3,
            border_radius=10,
        )
        pygame.draw.circle(
            self.screen, color if unlocked else (90, 90, 82), (x, y - 31), 8
        )
        if reached:
            pygame.draw.circle(self.screen, (250, 237, 151), (x, y - 31), 14, 2)

    def _draw_character(self, owner: str, x: float, y: float) -> None:
        x_i, feet_y = int(x), int(y)
        color = (252, 53, 28) if owner == "fire" else (34, 170, 246)
        light = (255, 178, 52) if owner == "fire" else (145, 232, 255)
        if owner == "fire":
            points = [
                (x_i, feet_y - 37),
                (x_i + 7, feet_y - 26),
                (x_i + 11, feet_y - 35),
                (x_i + 15, feet_y - 19),
                (x_i + 12, feet_y - 4),
                (x_i - 12, feet_y - 4),
                (x_i - 15, feet_y - 19),
                (x_i - 8, feet_y - 31),
            ]
        else:
            points = [
                (x_i, feet_y - 39),
                (x_i + 15, feet_y - 19),
                (x_i + 12, feet_y - 4),
                (x_i - 12, feet_y - 4),
                (x_i - 15, feet_y - 19),
            ]
        pygame.draw.polygon(self.screen, (16, 18, 16), points)
        inner = [
            (px + (x_i - px) * 0.12, py + (feet_y - 16 - py) * 0.08) for px, py in points
        ]
        pygame.draw.polygon(self.screen, color, inner)
        pygame.draw.circle(self.screen, light, (x_i - 5, feet_y - 20), 3)
        pygame.draw.circle(self.screen, light, (x_i + 5, feet_y - 20), 3)
        pygame.draw.circle(self.screen, (24, 28, 25), (x_i - 5, feet_y - 20), 1)
        pygame.draw.circle(self.screen, (24, 28, 25), (x_i + 5, feet_y - 20), 1)

    def _draw_vines(self) -> None:
        for x in range(30, self.W, 83):
            length = 12 + ((x + self.layout_seed) % 31)
            pygame.draw.line(self.screen, (17, 92, 35), (x, 0), (x + 5, length), 3)
            if x % 2:
                pygame.draw.ellipse(
                    self.screen, (25, 119, 43), (x + 1, length - 7, 12, 6)
                )

    def _draw_hud(self) -> None:
        overlay = pygame.Surface((self.W, 54), pygame.SRCALPHA)
        overlay.fill((5, 9, 8, 205))
        self.screen.blit(overlay, (0, 0))
        elapsed = self.steps / self.metadata["render_fps"]
        minutes, seconds = divmod(int(elapsed), 60)
        title = self._font(30).render(
            f"TEMPLE {self.layout_seed}     {minutes:02d}:{seconds:02d}",
            True,
            (237, 210, 109),
        )
        self.screen.blit(title, (self.W // 2 - title.get_width() // 2, 13))
        fire_text = self._font(23).render(
            f"FIRE  ◆ {int(self._gem_fraction('fire') * 3)}/3",
            True,
            (255, 93, 55),
        )
        water_text = self._font(23).render(
            f"WATER  ◆ {int(self._gem_fraction('water') * 3)}/3",
            True,
            (78, 199, 255),
        )
        switches = self._font(21).render(
            f"SWITCHES {sum(self.button_active.values())}/{len(self.button_active)}",
            True,
            (149, 224, 127),
        )
        self.screen.blit(water_text, (18, 17))
        self.screen.blit(fire_text, (self.W - fire_text.get_width() - 18, 17))
        self.screen.blit(switches, (18, self.H - 27))

    def close(self):
        self.window_closed = True
        if pygame is not None and self.screen is not None:
            pygame.display.quit()
        self.screen = None
        self.clock = None
