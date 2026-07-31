# firewater_env.py
import math
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from firewater.procedural_levels import generate_level

# Pygame is only needed for human rendering
try:
    import pygame
except ImportError:
    pygame = None


class FireWaterEnv(gym.Env):
    """
    Simple 2-character environment inspired by Fireboy & Watergirl.

    - 2 agents: Fire (red), Water (blue)
    - 1D movement along X with gravity & jumping in Y
    - Hazards: lava & water pools
    - Goals: two doors; both agents must reach their door to succeed
    - Discrete actions (0..6) controlling left/right/jump for one character at a time

    Legacy observation (19 floats, all in [-1, 1]):
      0: fire_x_norm
      1: fire_y_norm
      2: fire_vx_norm
      3: fire_vy_norm
      4: water_x_norm
      5: water_y_norm
      6: water_vx_norm
      7: water_vy_norm
      8: fire_on_ground (0/1)
      9: water_on_ground (0/1)
     10: fire_goal_x_norm
     11: water_goal_x_norm
     12: fire_to_goal_norm   (distance in x, normalized)
     13: water_to_goal_norm  (distance in x, normalized)
     14: min_goal_dist_norm
     15: same_side_flag (1 if both on same half of level, else 0)
     16: time_remaining_norm
     17: lava_center_x_norm
     18: water_center_x_norm

    Actions (Discrete(7)):
      0: no-op
      1: fire left
      2: fire right
      3: fire jump
      4: water left
      5: water right
      6: water jump

    Enhanced observations append goal heights, goal-completion flags, level id,
    six padded platform rectangles, and both complete hazard rectangles for a
    total of 56 floats. Generalized observations omit the level id and expose
    up to eight platform rectangles for a total of 64 floats, allowing one
    controller to act from geometry alone on procedural layouts. Legacy modes
    remain available so existing 19- and 48-input checkpoints still load.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    LEGACY_OBSERVATION_SIZE = 19
    MAX_PLATFORMS = 6
    GENERALIZED_MAX_PLATFORMS = 8
    ENHANCED_V1_OBSERVATION_SIZE = 48
    ENHANCED_OBSERVATION_SIZE = 56
    GENERALIZED_OBSERVATION_SIZE = 64

    def __init__(
        self,
        render_mode=None,
        level_id: int = 0,
        observation_mode: str = "legacy",
        procedural: bool = False,
        level_seed: int | None = None,
        procedural_seed_range: tuple[int, int] | None = None,
    ):
        super().__init__()

        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected None, "
                f"'human', or 'rgb_array'."
            )
        if observation_mode not in (
            "legacy",
            "enhanced_v1",
            "enhanced",
            "generalized",
        ):
            raise ValueError(
                f"Unsupported observation_mode {observation_mode!r}; expected "
                "'legacy', 'enhanced_v1', 'enhanced', or 'generalized'."
            )
        self.observation_mode = observation_mode

        # --- world / physics params ---
        self.W = 800
        self.H = 400
        self.ground_y = 320
        self.gravity = 1000.0          # px / s^2
        self.move_speed = 200.0        # px / s
        self.jump_speed = -500.0       # px / s
        self.dt = 1.0 / 30.0           # 30 FPS

        # Which level layout to use
        self.level_id = level_id
        self.procedural = bool(procedural or level_seed is not None)
        self.level_seed = int(level_seed) if level_seed is not None else None
        if procedural_seed_range is not None:
            seed_low, seed_high = map(int, procedural_seed_range)
            if seed_low < 0 or seed_high <= seed_low:
                raise ValueError(
                    "procedural_seed_range must be a non-negative [low, high) range"
                )
            self.procedural_seed_range = (seed_low, seed_high)
        else:
            self.procedural_seed_range = (0, int(np.iinfo(np.int32).max))
        self.layout_seed = None

        # episode control
        self.max_steps = 300
        self.steps = 0
        self._episode_done = False

        # character state
        self.fire_x = None
        self.fire_y = None
        self.fire_vx = None
        self.fire_vy = None
        self.fire_on_ground = None

        self.water_x = None
        self.water_y = None
        self.water_vx = None
        self.water_vy = None
        self.water_on_ground = None

        # doors / goals (set per-level)
        self.fire_goal_x = None
        self.fire_goal_y = None
        self.water_goal_x = None
        self.water_goal_y = None

        # hazards: (x1, y1, x2, y2) - set per-level
        self.lava_rect = None
        self.water_rect = None

        # raised platforms: list of rects (x1, y1, x2, y2)
        self.platforms = []

        # render / collision radius for characters
        self.char_radius = 12


        # goal flags (latched when a character reaches its door)
        self.fire_at_goal = False
        self.water_at_goal = False

        # per-step flags for when a door is just reached (for one-time bonuses)
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False

        # gym spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape={
                "legacy": (self.LEGACY_OBSERVATION_SIZE,),
                "enhanced_v1": (self.ENHANCED_V1_OBSERVATION_SIZE,),
                "enhanced": (self.ENHANCED_OBSERVATION_SIZE,),
                "generalized": (self.GENERALIZED_OBSERVATION_SIZE,),
            }[observation_mode],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(7)

        # rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_closed = False

        # initialize all level configs
        self._init_levels()

    # ------------------------------------------------------------------ #
    #  Level definitions
    # ------------------------------------------------------------------ #
    def _init_levels(self):
        """
        Define multiple level layouts of varying difficulty.

        Each level config contains:
          - fire_start: (x, y)
          - water_start: (x, y)
          - fire_goal: (x, y)
          - water_goal: (x, y)
          - lava_rect: (x1, y1, x2, y2)
          - water_rect: (x1, y1, x2, y2)
          - platforms: list of (x1, y1, x2, y2) for raised platforms.
        """
        gy = self.ground_y
        H = self.H
        W = self.W

        self.levels = []

        # Level 0: original simple horizontal layout (easiest)
        self.levels.append(
            {
                "fire_start": (W * 0.2, gy),
                "water_start": (W * 0.8, gy),
                "fire_goal": (W * 0.8, gy),
                "water_goal": (W * 0.2, gy),
                "lava_rect": (W * 0.42, gy, W * 0.49, H),
                "water_rect": (W * 0.53, gy, W * 0.60, H),
                "platforms": [],
            }
        )

        # Level 1: slightly wider pools, longer run-up (still mostly horizontal)
        self.levels.append(
            {
                "fire_start": (W * 0.15, gy),
                "water_start": (W * 0.85, gy),
                "fire_goal": (W * 0.85, gy),
                "water_goal": (W * 0.15, gy),
                "lava_rect": (W * 0.40, gy, W * 0.50, H),
                "water_rect": (W * 0.52, gy, W * 0.62, H),
                "platforms": [
                    # small mid-air ledge to force more controlled jumps
                    (W * 0.30, gy - 50, W * 0.36, gy - 40),
                    (W * 0.64, gy - 50, W * 0.70, gy - 40),
                ],
            }
        )

        # Level 2: doors on low raised platforms (must use platforms to get up)
        plat_y = gy - 80  # a bit higher, so door isn't trivial from ground
        self.levels.append(
            {
                "fire_start": (W * 0.2, gy),
                "water_start": (W * 0.8, gy),
                "fire_goal": (W * 0.78, plat_y),
                "water_goal": (W * 0.22, plat_y),
                "lava_rect": (W * 0.42, gy, W * 0.49, H),
                "water_rect": (W * 0.53, gy, W * 0.60, H),
                "platforms": [
                    # narrow door platforms (harder to land on)
                    (W * 0.72, plat_y, W * 0.84, plat_y + 10),  # fire door platform
                    (W * 0.16, plat_y, W * 0.28, plat_y + 10),  # water door platform
                    # small central step just above ground to help approach
                    (W * 0.48, gy - 40, W * 0.56, gy - 30),
                ],
            }
        )

        # Level 3: more vertical navigation, multiple steps to reach doors
        mid_y = gy - 70
        upper_y = gy - 120
        door_y = gy - 90
        self.levels.append(
            {
                "fire_start": (W * 0.15, gy),
                "water_start": (W * 0.85, gy),
                "fire_goal": (W * 0.80, door_y),
                "water_goal": (W * 0.20, door_y),
                "lava_rect": (W * 0.38, gy, W * 0.48, H),
                "water_rect": (W * 0.52, gy, W * 0.62, H),
                "platforms": [
                    # lower mid platforms
                    (W * 0.26, mid_y, W * 0.38, mid_y + 10),
                    (W * 0.62, mid_y, W * 0.74, mid_y + 10),
                    # upper stepping stones near center
                    (W * 0.44, upper_y, W * 0.52, upper_y + 10),
                    # door platforms (narrow)
                    (W * 0.74, door_y, W * 0.86, door_y + 10),
                    (W * 0.14, door_y, W * 0.26, door_y + 10),
                ],
            }
        )

        # Level 4: hardest – higher doors and more precise platforming
        # A single jump from ground can't reach the doors; you MUST use platforms.
        high_y = gy - 140
        mid_y2 = gy - 90
        low_y2 = gy - 50
        self.levels.append(
            {
                "fire_start": (W * 0.1, gy),
                "water_start": (W * 0.9, gy),
                "fire_goal": (W * 0.82, high_y),
                "water_goal": (W * 0.18, high_y),
                "lava_rect": (W * 0.40, gy, W * 0.47, H),
                "water_rect": (W * 0.53, gy, W * 0.60, H),
                "platforms": [
                    # central lower stepping platforms
                    (W * 0.30, low_y2, W * 0.38, low_y2 + 10),
                    (W * 0.62, low_y2, W * 0.70, low_y2 + 10),
                    # mid-height stepping platforms
                    (W * 0.42, mid_y2, W * 0.50, mid_y2 + 10),
                    (W * 0.50, mid_y2 - 30, W * 0.58, mid_y2 - 20),
                    # high door platforms (narrow, hard)
                    (W * 0.76, high_y, W * 0.88, high_y + 10),
                    (W * 0.12, high_y, W * 0.24, high_y + 10),
                ],
            }
        )

    def _load_level(self, options=None):
        """Apply the current level configuration to the environment."""
        if self.procedural:
            requested_seed = (
                options.get("level_seed")
                if options is not None and "level_seed" in options
                else self.level_seed
            )
            if requested_seed is None:
                seed_low, seed_high = self.procedural_seed_range
                requested_seed = int(
                    self.np_random.integers(seed_low, seed_high)
                )
            self.layout_seed = int(requested_seed)
            cfg = generate_level(
                self.layout_seed,
                width=self.W,
                height=self.H,
                ground_y=self.ground_y,
            ).as_env_config()
            self.level_id = -1
        else:
            self.layout_seed = None
            if not hasattr(self, "levels") or len(self.levels) == 0:
                self._init_levels()

            idx = int(self.level_id)
            if not 0 <= idx < len(self.levels):
                raise ValueError(
                    f"Unknown level_id {self.level_id!r}; expected an integer from "
                    f"0 to {len(self.levels) - 1}."
                )
            self.level_id = idx
            cfg = self.levels[idx]

        self.fire_goal_x, self.fire_goal_y = cfg["fire_goal"]
        self.water_goal_x, self.water_goal_y = cfg["water_goal"]
        self.lava_rect = cfg["lava_rect"]
        self.water_rect = cfg["water_rect"]
        self.platforms = list(cfg.get("platforms", []))

        # starting positions used in reset()
        self._fire_start = cfg["fire_start"]
        self._water_start = cfg["water_start"]

    # ------------------------------------------------------------------ #
    #  Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # load/update level config
        self._load_level(options=options)

        # spawn characters from level start positions
        fire_start_x, fire_start_y = self._fire_start
        water_start_x, water_start_y = self._water_start

        self.fire_x = fire_start_x
        self.fire_y = fire_start_y
        self.fire_vx = 0.0
        self.fire_vy = 0.0
        self.fire_on_ground = True

        self.water_x = water_start_x
        self.water_y = water_start_y
        self.water_vx = 0.0
        self.water_vy = 0.0
        self.water_on_ground = True

        self.steps = 0
        self._episode_done = False

        # reset goal flags
        self.fire_at_goal = False
        self.water_at_goal = False

        # reset per-step goal latch flags
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False

        obs = self._get_obs()
        info = {
            "level_id": self.level_id,
            "layout_seed": self.layout_seed,
            "procedural": self.procedural,
        }

        if self.render_mode == "human":
            self._ensure_render()

        return obs, info

    def step(self, action):
        """Single environment step (RL mode).

        Reward design focused on reaching both doors with dense shaping:

        - Small per-step negative reward to encourage finishing sooner.
        - Progress shaping based on change in horizontal distance of both
          characters to their own doors (positive for moving closer,
          negative for moving away).
        - Bonuses when each character reaches its door.
        - Large bonus when both doors are reached (success).
        - Penalty for dying in the wrong pool.
        - Penalty for timing out, scaled by remaining distance to doors.
        """

        if self._episode_done:
            raise RuntimeError("step() called after episode end; call reset() first")

        action = int(action)

        # 0) Combined 2D distance to both doors. Using the team total matters
        # after one character reaches its door: min(distance) would be zero and
        # would stop shaping progress for the remaining character entirely.
        team_goal_before = self._compute_team_goal_dist()

        # 1) Distances to doors *before* action (normalized by level width)
        fire_dist_before = abs(self.fire_x - self.fire_goal_x) / self.W
        water_dist_before = abs(self.water_x - self.water_goal_x) / self.W

        # 2) Apply action and physics
        self._apply_action(action)
        self._physics_step()
        self.steps += 1

        # 3) Distances *after* physics
        fire_dist_after = abs(self.fire_x - self.fire_goal_x) / self.W
        water_dist_after = abs(self.water_x - self.water_goal_x) / self.W

        # 3b) Combined 2D distance after physics.
        team_goal_after = self._compute_team_goal_dist()
        diag = math.hypot(self.W, self.H)
        if diag > 0.0:
            delta_team_goal = (team_goal_before - team_goal_after) / (2.0 * diag)
            # Clamp to avoid rare large jumps dominating the reward
            delta_team_goal = float(np.clip(delta_team_goal, -0.25, 0.25))
        else:
            delta_team_goal = 0.0

        # ---------------- Base reward: small negative per step ----------------
        # Encourages shorter episodes once the agent knows how to succeed.
        reward = -0.005

        # ---------------- Progress shaping (both characters) ----------------
        fire_progress = fire_dist_before - fire_dist_after
        water_progress = water_dist_before - water_dist_after
        total_progress = fire_progress + water_progress

        # Clamp for stability against rare large jumps.
        total_progress = float(np.clip(total_progress, -0.25, 0.25))

        # Reward both positive and negative progress so going backwards
        # is clearly worse than standing still.
        reward += 1.5 * total_progress

        # Additional shaping rewards true 2D team progress so vertical/platform
        # movement and the second character remain useful learning signals.
        reward += 2.0 * delta_team_goal

        # ---------------- Door bonuses (per-character) ----------------
        per_door_bonus = 10.0
        if self.just_reached_fire_goal:
            reward += per_door_bonus
        if self.just_reached_water_goal:
            reward += per_door_bonus

        # ---------------- Episode termination logic ----------------
        success = self._both_at_goals()
        dead = self._fell_into_wrong_pool()
        timeout = self.steps >= self.max_steps

        terminated = bool(success or dead)
        truncated = bool(timeout and not terminated)

        info = {
            "level_id": self.level_id,
            "layout_seed": self.layout_seed,
            "procedural": self.procedural,
            "steps": self.steps,
            "success": bool(success),
            "dead": bool(dead),
            "timeout": bool(timeout),
            "fire_at_goal": bool(self.fire_at_goal),
            "water_at_goal": bool(self.water_at_goal),
            "reason": None,
        }

        # ---------------- Terminal bonuses / penalties ----------------
        if success:
            # Main objective: both characters at their doors.
            reward += 60.0
            info["reason"] = "success"
        elif dead:
            # Preserve enough exploration pressure to learn the first safe
            # crossings, but close the exploit where a +10 door bonus nearly
            # cancels an immediate death by the remaining character.
            reward -= 10.0
            if self.fire_at_goal != self.water_at_goal:
                reward -= 15.0
            info["reason"] = "hazard"
        elif timeout:
            # Penalize more if still far from doors.
            mean_dist = 0.5 * (fire_dist_after + water_dist_after)
            reward -= 3.0 + 6.0 * mean_dist
            info["reason"] = "timeout"

        self._episode_done = terminated or truncated

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    def manual_step(self, fire_cmd, water_cmd):
        """
        Step the environment using simple string commands for each character.

        fire_cmd / water_cmd ∈ {
            None,
            "left", "right", "jump",
            "left_jump", "right_jump"   # move + jump on same frame
        }.

        - Left/right work both on the ground and in the air.
        - Jumps only trigger when the character is on the ground.
        - This is ONLY for human play; RL still uses step(action).
        """

        # ---------------- Fire controls ----------------
        if not self.fire_at_goal:
            # Decode combined commands
            fire_move = 0   # -1 = left, +1 = right, 0 = none
            fire_wants_jump = False

            if fire_cmd in ("left", "left_jump"):
                fire_move = -1
            elif fire_cmd in ("right", "right_jump"):
                fire_move = 1

            if fire_cmd in ("jump", "left_jump", "right_jump"):
                fire_wants_jump = True

            # horizontal movement allowed even in air
            self.fire_vx = 0.0
            if fire_move == -1:
                self.fire_vx = -self.move_speed
            elif fire_move == 1:
                self.fire_vx = self.move_speed

            # jump only if on ground
            if fire_wants_jump and self.fire_on_ground:
                self.fire_vy = self.jump_speed
        else:
            # stay pinned at door
            self.fire_vx = 0.0

        # ---------------- Water controls ----------------
        if not self.water_at_goal:
            water_move = 0
            water_wants_jump = False

            if water_cmd in ("left", "left_jump"):
                water_move = -1
            elif water_cmd in ("right", "right_jump"):
                water_move = 1

            if water_cmd in ("jump", "left_jump", "right_jump"):
                water_wants_jump = True

            # horizontal movement allowed even in air
            self.water_vx = 0.0
            if water_move == -1:
                self.water_vx = -self.move_speed
            elif water_move == 1:
                self.water_vx = self.move_speed

            # jump only if on ground
            if water_wants_jump and self.water_on_ground:
                self.water_vy = self.jump_speed
        else:
            # stay pinned at door
            self.water_vx = 0.0

        # Physics & game logic (no RL reward here)
        self._physics_step()
        self.steps += 1

    def render(self):
        if self.render_mode == "human":
            self._ensure_render()
            self._render_frame()
        elif self.render_mode == "rgb_array":
            self._ensure_render()
            return self._render_frame(return_array=True)

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None
        self.window_closed = True

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _update_goal_states(self):
        """Latch characters at their doors once they reach them.

        We now require both horizontal *and* vertical proximity to the door
        so that agents can't trigger the door while running underneath a
        raised platform.
        """
        tol_x = 25.0  # horizontal tolerance around the door center
        tol_y = 15.0  # vertical tolerance around the door base

        # reset per-step latching flags
        self.just_reached_fire_goal = False
        self.just_reached_water_goal = False

        # Fire door
        if (
            not self.fire_at_goal
            and self.fire_on_ground
            and abs(self.fire_x - self.fire_goal_x) < tol_x
            and abs(self.fire_y - self.fire_goal_y) < tol_y
        ):
            self.fire_at_goal = True
            self.just_reached_fire_goal = True
            # snap to door and stop moving
            self.fire_x = self.fire_goal_x
            self.fire_y = self.fire_goal_y
            self.fire_vx = 0.0
            self.fire_vy = 0.0

        # Water door
        if (
            not self.water_at_goal
            and self.water_on_ground
            and abs(self.water_x - self.water_goal_x) < tol_x
            and abs(self.water_y - self.water_goal_y) < tol_y
        ):
            self.water_at_goal = True
            self.just_reached_water_goal = True
            # snap to door and stop moving
            self.water_x = self.water_goal_x
            self.water_y = self.water_goal_y
            self.water_vx = 0.0
            self.water_vy = 0.0

    def _compute_min_goal_dist(self):
        fire_pos = np.array([self.fire_x, self.fire_y], dtype=np.float32)
        fire_goal = np.array([self.fire_goal_x, self.fire_goal_y], dtype=np.float32)

        water_pos = np.array([self.water_x, self.water_y], dtype=np.float32)
        water_goal = np.array([self.water_goal_x, self.water_goal_y], dtype=np.float32)

        fire_dist = np.linalg.norm(fire_pos - fire_goal)
        water_dist = np.linalg.norm(water_pos - water_goal)

        return float(min(fire_dist, water_dist))

    def _compute_team_goal_dist(self):
        """Return the sum of both characters' Euclidean door distances."""
        fire_dist = math.hypot(
            self.fire_x - self.fire_goal_x,
            self.fire_y - self.fire_goal_y,
        )
        water_dist = math.hypot(
            self.water_x - self.water_goal_x,
            self.water_y - self.water_goal_y,
        )
        return float(fire_dist + water_dist)


    def _both_at_goals(self, tol=35.0):
        _ = tol  # unused, kept for compatibility
        return self.fire_at_goal and self.water_at_goal

    def _fell_into_wrong_pool(self):
        def in_rect_circle(x, y, rect, r):
            """Approximate collision of a circle (character) with an axis-aligned rect.

            Character center is (x, y), radius r. This makes standing "on" the pool
            visually line up with actually dying.
            """
            x1, y1, x2, y2 = rect
            return (x1 - r <= x <= x2 + r) and (y1 - r <= y <= y2)

        r = getattr(self, "char_radius", 12)

        # fire dies in water
        fire_in_water = in_rect_circle(self.fire_x, self.fire_y, self.water_rect, r)
        # water dies in lava
        water_in_lava = in_rect_circle(self.water_x, self.water_y, self.lava_rect, r)

        return fire_in_water or water_in_lava

    def _near_deadly_pool(self, who: str, margin: float = 90.0) -> bool:
        """Return True if the specified character is within a horizontal window
        around its *deadly* pool, where a jump is actually useful.
        """
        lava_x1, _, lava_x2, _ = self.lava_rect
        water_x1, _, water_x2, _ = self.water_rect

        if who == "fire":
            x = self.fire_x
            # Fire dies in WATER, so only consider the water pool.
            return (water_x1 - margin) <= x <= (water_x2 + margin)
        elif who == "water":
            x = self.water_x
            # Water dies in LAVA, so only consider the lava pool.
            return (lava_x1 - margin) <= x <= (lava_x2 + margin)
        else:
            return False

    def _apply_action(self, action: int):
        """
        Discrete actions (0..6):

          0: no-op
          1: fire left
          2: fire right
          3: fire jump  (preserves horizontal velocity)
          4: water left
          5: water right
          6: water jump (preserves horizontal velocity)
        """

        # ---------------- Fire controls ----------------
        if not self.fire_at_goal:
            # Only change fire_vx if we actually press a horizontal button.
            if action == 1:          # fire left
                self.fire_vx = -self.move_speed
            elif action == 2:        # fire right
                self.fire_vx = self.move_speed
            # Jump whenever on ground (needed for platforming and doors).
            if action == 3 and self.fire_on_ground:
                self.fire_vy = self.jump_speed
        else:
            # Stay pinned at the door once reached
            self.fire_vx = 0.0

        # ---------------- Water controls ----------------
        if not self.water_at_goal:
            # Only change water_vx if we actually press a horizontal button.
            if action == 4:          # water left
                self.water_vx = -self.move_speed
            elif action == 5:        # water right
                self.water_vx = self.move_speed
            # Jump whenever on ground (needed for platforming and doors).
            if action == 6 and self.water_on_ground:
                self.water_vy = self.jump_speed
        else:
            # Stay pinned at the door once reached
            self.water_vx = 0.0

    def _physics_step(self):
        # store previous positions for platform collision
        old_fire_x, old_fire_y = self.fire_x, self.fire_y
        old_water_x, old_water_y = self.water_x, self.water_y

        # apply gravity
        self.fire_vy += self.gravity * self.dt
        self.water_vy += self.gravity * self.dt

        # integrate
        self.fire_x += self.fire_vx * self.dt
        self.fire_y += self.fire_vy * self.dt
        self.water_x += self.water_vx * self.dt
        self.water_y += self.water_vy * self.dt

        # clamp to screen / ground / platforms (platforms are now SOLID: top and bottom)
        def clamp_char(x, y, vx, vy, old_x, old_y):
            # light horizontal friction
            vx *= 0.9
            if abs(vx) < 1.0:  # snap to zero when very small
                vx = 0.0

            # horizontal bounds
            x = max(0.0, min(self.W, x))

            on_ground = False

            # platform collisions
            # Treat platforms as solid rectangles: you can't jump up through them.
            for (px1, py1, px2, py2) in getattr(self, "platforms", []):
                top = py1
                bottom = py2

                # Hitting the TOP surface (landing from above, moving down)
                if vy >= 0 and old_y <= top <= y and px1 <= x <= px2:
                    y = top
                    vy = 0.0
                    on_ground = True
                    # Once we land on something, we don't process other platforms
                    break

                # Hitting the BOTTOM surface (jumping up from below, moving up)
                if vy < 0 and old_y >= bottom >= y and px1 <= x <= px2:
                    # Bonk head on underside and fall back down
                    y = bottom
                    vy = 0.0
                    # Not on_ground, just stopped upward motion
                    break

            # ground collision
            if y >= self.ground_y:
                y = self.ground_y
                vy = 0.0
                on_ground = True

            # ceiling
            if y < 0:
                y = 0.0
                vy = 0.0

            return x, y, vx, vy, on_ground

        (self.fire_x, self.fire_y, self.fire_vx, self.fire_vy, self.fire_on_ground) = \
            clamp_char(self.fire_x, self.fire_y, self.fire_vx, self.fire_vy, old_fire_x, old_fire_y)

        (self.water_x, self.water_y, self.water_vx, self.water_vy, self.water_on_ground) = \
            clamp_char(self.water_x, self.water_y, self.water_vx, self.water_vy, old_water_x, old_water_y)

        # After applying physics and ground/platform collisions, latch at goals if reached
        self._update_goal_states()

    def _get_obs(self):
        # normalize positions and velocities to roughly [-1, 1]
        fire_x_norm = (self.fire_x / self.W) * 2.0 - 1.0
        fire_y_norm = (self.fire_y / self.H) * 2.0 - 1.0
        water_x_norm = (self.water_x / self.W) * 2.0 - 1.0
        water_y_norm = (self.water_y / self.H) * 2.0 - 1.0

        vx_scale = self.move_speed
        vy_scale = abs(self.jump_speed)

        fire_vx_norm = np.clip(self.fire_vx / vx_scale, -1.0, 1.0)
        fire_vy_norm = np.clip(self.fire_vy / vy_scale, -1.0, 1.0)
        water_vx_norm = np.clip(self.water_vx / vx_scale, -1.0, 1.0)
        water_vy_norm = np.clip(self.water_vy / vy_scale, -1.0, 1.0)

        fire_goal_x_norm = (self.fire_goal_x / self.W) * 2.0 - 1.0
        water_goal_x_norm = (self.water_goal_x / self.W) * 2.0 - 1.0

        fire_to_goal = abs(self.fire_x - self.fire_goal_x) / self.W
        water_to_goal = abs(self.water_x - self.water_goal_x) / self.W
        min_goal_dist = min(fire_to_goal, water_to_goal)

        # hazard centers (lava for fire, water for water) in normalized X
        lava_x1, _, lava_x2, _ = self.lava_rect
        water_x1, _, water_x2, _ = self.water_rect
        lava_center_x = 0.5 * (lava_x1 + lava_x2)
        water_center_x = 0.5 * (water_x1 + water_x2)
        lava_center_x_norm = (lava_center_x / self.W) * 2.0 - 1.0
        water_center_x_norm = (water_center_x / self.W) * 2.0 - 1.0

        same_side = 1.0 if (self.fire_x < self.W / 2) == (self.water_x < self.W / 2) else 0.0

        time_remaining = 1.0 - (self.steps / self.max_steps)
        time_remaining = float(np.clip(time_remaining, 0.0, 1.0))

        values = [
                fire_x_norm,                 # 0
                fire_y_norm,                 # 1
                fire_vx_norm,                # 2
                fire_vy_norm,                # 3
                water_x_norm,                # 4
                water_y_norm,                # 5
                water_vx_norm,               # 6
                water_vy_norm,               # 7
                float(self.fire_on_ground),  # 8
                float(self.water_on_ground), # 9
                fire_goal_x_norm,            # 10
                water_goal_x_norm,           # 11
                float(np.clip(fire_to_goal, 0.0, 1.0)),   # 12
                float(np.clip(water_to_goal, 0.0, 1.0)),  # 13
                float(np.clip(min_goal_dist, 0.0, 1.0)),  # 14
                same_side,                   # 15
                time_remaining,              # 16
                lava_center_x_norm,          # 17
                water_center_x_norm,         # 18
            ]

        if self.observation_mode in ("enhanced_v1", "enhanced"):
            fire_goal_y_norm = (self.fire_goal_y / self.H) * 2.0 - 1.0
            water_goal_y_norm = (self.water_goal_y / self.H) * 2.0 - 1.0
            level_norm = (self.level_id / (len(self.levels) - 1)) * 2.0 - 1.0
            values.extend(
                [
                    fire_goal_y_norm,             # 19
                    water_goal_y_norm,            # 20
                    float(self.fire_at_goal),     # 21
                    float(self.water_at_goal),    # 22
                    level_norm,                   # 23
                ]
            )

            for platform_idx in range(self.MAX_PLATFORMS):
                if platform_idx < len(self.platforms):
                    px1, py1, px2, py2 = self.platforms[platform_idx]
                    values.extend(
                        [
                            (px1 / self.W) * 2.0 - 1.0,
                            (py1 / self.H) * 2.0 - 1.0,
                            (px2 / self.W) * 2.0 - 1.0,
                            (py2 / self.H) * 2.0 - 1.0,
                        ]
                    )
                else:
                    values.extend([-1.0, -1.0, -1.0, -1.0])

        if self.observation_mode == "enhanced":
            for x1, y1, x2, y2 in (self.lava_rect, self.water_rect):
                values.extend(
                    [
                        (x1 / self.W) * 2.0 - 1.0,
                        (y1 / self.H) * 2.0 - 1.0,
                        (x2 / self.W) * 2.0 - 1.0,
                        (y2 / self.H) * 2.0 - 1.0,
                    ]
                )

        if self.observation_mode == "generalized":
            fire_goal_y_norm = (self.fire_goal_y / self.H) * 2.0 - 1.0
            water_goal_y_norm = (self.water_goal_y / self.H) * 2.0 - 1.0
            values.extend(
                [
                    fire_goal_y_norm,
                    water_goal_y_norm,
                    float(self.fire_at_goal),
                    float(self.water_at_goal),
                    len(self.platforms) / self.GENERALIZED_MAX_PLATFORMS,
                ]
            )
            for platform_idx in range(self.GENERALIZED_MAX_PLATFORMS):
                if platform_idx < len(self.platforms):
                    px1, py1, px2, py2 = self.platforms[platform_idx]
                    values.extend(
                        [
                            (px1 / self.W) * 2.0 - 1.0,
                            (py1 / self.H) * 2.0 - 1.0,
                            (px2 / self.W) * 2.0 - 1.0,
                            (py2 / self.H) * 2.0 - 1.0,
                        ]
                    )
                else:
                    values.extend([-1.0, -1.0, -1.0, -1.0])
            for x1, y1, x2, y2 in (self.lava_rect, self.water_rect):
                values.extend(
                    [
                        (x1 / self.W) * 2.0 - 1.0,
                        (y1 / self.H) * 2.0 - 1.0,
                        (x2 / self.W) * 2.0 - 1.0,
                        (y2 / self.H) * 2.0 - 1.0,
                    ]
                )

        obs = np.asarray(values, dtype=np.float32)

        return obs

    # ------------------------------------------------------------------ #
    #  Rendering helpers
    # ------------------------------------------------------------------ #
    def _ensure_render(self):
        if pygame is None:
            raise RuntimeError("pygame must be installed to render FireWaterEnv")

        if self.window_closed:
            return False

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Fire & Water RL Env")
                self.screen = pygame.display.set_mode((self.W, self.H))
                self.clock = pygame.time.Clock()
            else:
                # rgb_array rendering does not need a display or event queue.
                self.screen = pygame.Surface((self.W, self.H))
        return True

    def _render_frame(self, return_array=False):
        if pygame is None:
            raise RuntimeError("pygame must be installed to render FireWaterEnv")

        if not self._ensure_render():
            return None

        # Only a real display has a window event queue. If the user closes the
        # window, stop this frame immediately instead of drawing to a surface
        # that close() has just discarded.
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self.screen.fill((0, 0, 0))

        # draw ground
        pygame.draw.rect(
            self.screen,
            (80, 80, 80),
            (0, self.ground_y, self.W, self.H - self.ground_y),
        )

        # lava (red) and water (blue) pools
        lava_x1, lava_y1, lava_x2, lava_y2 = self.lava_rect
        water_x1, water_y1, water_x2, water_y2 = self.water_rect

        pygame.draw.rect(
            self.screen,
            (255, 80, 0),
            (lava_x1, lava_y1, lava_x2 - lava_x1, lava_y2 - lava_y1),
        )
        pygame.draw.rect(
            self.screen,
            (0, 120, 255),
            (water_x1, water_y1, water_x2 - water_x1, water_y2 - water_y1),
        )

        # platforms (raised ground)
        for (px1, py1, px2, py2) in getattr(self, "platforms", []):
            pygame.draw.rect(
                self.screen,
                (120, 120, 120),
                (px1, py1, px2 - px1, py2 - py1),
            )

        # doors (colored by character)
        door_w = 30
        door_h = 60
        # Fire door: reddish
        pygame.draw.rect(
            self.screen,
            (255, 80, 80),
            (self.fire_goal_x - door_w / 2, self.fire_goal_y - door_h, door_w, door_h),
        )
        # Water door: bluish
        pygame.draw.rect(
            self.screen,
            (80, 160, 255),
            (self.water_goal_x - door_w / 2, self.water_goal_y - door_h, door_w, door_h),
        )

        # characters
        radius = getattr(self, "char_radius", 12)
        pygame.draw.circle(
            self.screen, (255, 50, 50), (int(self.fire_x), int(self.fire_y - radius)), radius
        )
        pygame.draw.circle(
            self.screen, (50, 150, 255), (int(self.water_x), int(self.water_y - radius)), radius
        )

        if self.render_mode == "human":
            pygame.display.flip()

            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])

        if return_array:
            # array3d returns (width, height, channels); Gymnasium expects
            # (height, width, channels). Copy so callers never hold a lock on
            # the underlying Pygame surface.
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))


# ------------------------------------------------------------------ #
#  Demo recording helpers (for behavior cloning)
# ------------------------------------------------------------------ #
def _keyboard_to_action():
    """
    Map keyboard input to a discrete action for the RL agent.

    Controls (when using record_demo):
      - Fire (red):  LEFT/RIGHT/UP arrows
          LEFT  -> action 1 (fire left)
          RIGHT -> action 2 (fire right)
          UP    -> action 3 (fire jump)
      - Water (blue): A/D/W keys
          A -> action 4 (water left)
          D -> action 5 (water right)
          W -> action 6 (water jump)

    If multiple keys are pressed, fire controls have priority over water.
    If no movement key is pressed, action 0 (no-op) is used.

    This is only used for collecting demonstration trajectories with
    record_demo(); standard RL training still uses env.step(action)
    normally.
    """
    if pygame is None:
        raise RuntimeError("pygame must be installed for keyboard control demos")

    # Query current key state. This does not consume events, so it is
    # safe to use alongside the event loop in _render_frame.
    keys = pygame.key.get_pressed()

    # Allow ESC to act as "no-op but you can also use it to exit the demo loop"
    # (the actual exit check is handled in record_demo).
    # Fire controls have priority
    if keys[pygame.K_LEFT]:
        return 1  # fire left
    if keys[pygame.K_RIGHT]:
        return 2  # fire right
    if keys[pygame.K_UP]:
        return 3  # fire jump

    # Water controls
    if keys[pygame.K_a]:
        return 4  # water left
    if keys[pygame.K_d]:
        return 5  # water right
    if keys[pygame.K_w]:
        return 6  # water jump

    # Default: no-op
    return 0


def record_demo(
    level_id: int = 0,
    max_steps: int = 500,
    out_path: str = "demo_level0_run1.npz",
    observation_mode: str = "enhanced",
):
    """
    Play a level manually and record a demonstration trajectory for behavior cloning.

    Usage (from a separate script or an interactive session):
      >>> from firewater_env import record_demo
      >>> record_demo(level_id=0, out_path="demo_level0_run1.npz")

    You control the characters with the keyboard:
      - Fire (red):  LEFT / RIGHT / UP arrows
      - Water (blue): A / D / W keys

    The function saves an .npz file with two arrays:
      - obs:  (T, obs_dim)
      - acts: (T,) of discrete actions in {0..6}
    """
    if pygame is None:
        raise RuntimeError("pygame must be installed to record demos")

    env = FireWaterEnv(
        render_mode="human",
        level_id=level_id,
        observation_mode=observation_mode,
    )
    obs, info = env.reset()

    obs_list = []
    act_list = []
    reward_list = []

    done = False
    truncated = False
    step_idx = 0

    while not (done or truncated) and step_idx < max_steps:
        # Allow user to close the window or press ESC to end the recording.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                truncated = True
                break

        if done:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            # Gracefully stop recording without forcing an episode termination
            truncated = True
            break

        action = _keyboard_to_action()

        obs_list.append(obs)
        act_list.append(int(action))

        obs, reward, done, truncated, info = env.step(action)
        reward_list.append(reward)
        step_idx += 1

    env.close()

    if len(obs_list) == 0:
        print("No steps recorded; demo not saved.")
        return

    obs_arr = np.stack(obs_list, axis=0)
    acts_arr = np.array(act_list, dtype=np.int64)
    rewards_arr = np.array(reward_list, dtype=np.float32)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        obs=obs_arr,
        acts=acts_arr,
        rewards=rewards_arr,
        level_id=np.asarray(level_id, dtype=np.int64),
        reason=np.asarray(info.get("reason") or "interrupted"),
        observation_mode=np.asarray(observation_mode),
        format_version=np.asarray(1, dtype=np.int64),
    )
    print(
        f"Saved demo with {len(obs_list)} steps and return "
        f"{rewards_arr.sum():.3f} to {destination}"
    )
