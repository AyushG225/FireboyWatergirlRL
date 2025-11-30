
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


class FireWaterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        
        self.W = 800
        self.H = 400
        self.ground_y = 320
        self.gravity = 1000.0          
        self.move_speed = 200.0        
        self.jump_speed = -500.0       
        self.dt = 1.0 / 30.0           

        
        self.max_steps = 300
        self.steps = 0

        
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

        
        self.fire_goal_x = self.W * 0.8
        self.fire_goal_y = self.ground_y
        self.water_goal_x = self.W * 0.2
        self.water_goal_y = self.ground_y

        
        self.lava_rect = (self.W * 0.42, self.ground_y, self.W * 0.49, self.H)
        self.water_rect = (self.W * 0.53, self.ground_y, self.W * 0.60, self.H)

        
        self.char_radius = 12

        
        self.prev_team_x_dist = None
        
        self.fire_crossed_water = False
        self.water_crossed_lava = False

        
        self.fire_at_goal = False
        self.water_at_goal = False

        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(7)

        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        
        self.fire_x = self.W * 0.2
        self.fire_y = self.ground_y
        self.fire_vx = 0.0
        self.fire_vy = 0.0
        self.fire_on_ground = True

        self.water_x = self.W * 0.8
        self.water_y = self.ground_y
        self.water_vx = 0.0
        self.water_vy = 0.0
        self.water_on_ground = True

        self.steps = 0
        self.prev_team_x_dist = self._compute_team_x_dist()

        
        self.fire_crossed_water = False
        self.water_crossed_lava = False

        
        self.fire_at_goal = False
        self.water_at_goal = False

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._ensure_render()

        return obs, info

    def step(self, action):
        old_fire_to_goal = abs(self.fire_x - self.fire_goal_x) / self.W
        old_water_to_goal = abs(self.water_x - self.water_goal_x) / self.W

        
        self._apply_action(int(action))
        self._physics_step()
        self.steps += 1

        
        new_fire_to_goal = abs(self.fire_x - self.fire_goal_x) / self.W
        new_water_to_goal = abs(self.water_x - self.water_goal_x) / self.W

        delta_fire = old_fire_to_goal - new_fire_to_goal   
        delta_water = old_water_to_goal - new_water_to_goal  

        
        
        if action in (1, 2, 3):
            delta_used = delta_fire
        elif action in (4, 5, 6):
            delta_used = delta_water
        else:  
            delta_used = 0.5 * (delta_fire + delta_water)

        
        delta_used = float(np.clip(delta_used, -0.05, 0.05))

        
        reward = -0.0005  

        
        
        
        reward += 5.0 * delta_used

        
        
        
        water_x1, _, water_x2, _ = self.water_rect
        lava_x1,  _, lava_x2,  _ = self.lava_rect

        if (
            not self.fire_crossed_water
            and self.fire_on_ground
            and (self.fire_x > water_x2)
            and not self._fell_into_wrong_pool()
        ):
            reward += 10.0
            self.fire_crossed_water = True

        
        
        if (
            not self.water_crossed_lava
            and self.water_on_ground
            and (self.water_x < lava_x1)
            and not self._fell_into_wrong_pool()
        ):
            reward += 10.0
            self.water_crossed_lava = True

        
        success = self._both_at_goals()
        dead = self._fell_into_wrong_pool()
        timeout = self.steps >= self.max_steps

        terminated = bool(success or dead)
        truncated = bool(timeout and not terminated)

        info = {
            "success": bool(success),
            "reason": None,
        }

        
        
        if success:
            reward += 40.0
            info["reason"] = "success"
        elif dead:
            reward -= 20.0
            info["reason"] = "hazard"
        elif timeout:
            reward -= 5.0
            info["reason"] = "timeout"

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        
        self.prev_team_x_dist = float(new_fire_to_goal + new_water_to_goal)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._ensure_render()
            self._render_frame()
        elif self.render_mode == "rgb_array":
            
            return self._render_frame(return_array=True)

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self.screen = None
        self.clock = None

    
    
    
    def _update_goal_states(self):
        tol_x = 25.0  

        
        if (
            not self.fire_at_goal
            and self.fire_on_ground
            and abs(self.fire_x - self.fire_goal_x) < tol_x
        ):
            self.fire_at_goal = True
            
            self.fire_x = self.fire_goal_x
            self.fire_vx = 0.0
            self.fire_vy = 0.0

        
        if (
            not self.water_at_goal
            and self.water_on_ground
            and abs(self.water_x - self.water_goal_x) < tol_x
        ):
            self.water_at_goal = True
            
            self.water_x = self.water_goal_x
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

    def _compute_team_x_dist(self):
        fire_to_goal = abs(self.fire_x - self.fire_goal_x) / self.W
        water_to_goal = abs(self.water_x - self.water_goal_x) / self.W
        return float(fire_to_goal + water_to_goal)

    def _both_at_goals(self, tol=35.0):
        _ = tol  
        return self.fire_at_goal and self.water_at_goal

    def _fell_into_wrong_pool(self):
        def in_rect_circle(x, y, rect, r):
            x1, y1, x2, y2 = rect
            
            return (x1 - r <= x <= x2 + r) and (y1 - r <= y <= y2)

        r = getattr(self, "char_radius", 12)

        
        fire_in_water = in_rect_circle(self.fire_x, self.fire_y, self.water_rect, r)
        
        water_in_lava = in_rect_circle(self.water_x, self.water_y, self.lava_rect, r)

        return fire_in_water or water_in_lava

    def _near_deadly_pool(self, who: str, margin: float = 90.0) -> bool:
        lava_x1, _, lava_x2, _ = self.lava_rect
        water_x1, _, water_x2, _ = self.water_rect

        if who == "fire":
            x = self.fire_x
            
            return (water_x1 - margin) <= x <= (water_x2 + margin)
        elif who == "water":
            x = self.water_x
            
            return (lava_x1 - margin) <= x <= (lava_x2 + margin)
        else:
            return False

    def _apply_action(self, action: int):
        if not self.fire_at_goal:
            
            if action == 1:          
                self.fire_vx = -self.move_speed
            elif action == 2:        
                self.fire_vx = self.move_speed
            
            
            
            if action == 3 and self.fire_on_ground and self._near_deadly_pool("fire"):
                self.fire_vy = self.jump_speed
        else:
            
            self.fire_vx = 0.0

        
        if not self.water_at_goal:
            
            if action == 4:          
                self.water_vx = -self.move_speed
            elif action == 5:        
                self.water_vx = self.move_speed
            
            
            
            if action == 6 and self.water_on_ground and self._near_deadly_pool("water"):
                self.water_vy = self.jump_speed
        else:
            
            self.water_vx = 0.0

    def _physics_step(self):
        
        self.fire_vy += self.gravity * self.dt
        self.water_vy += self.gravity * self.dt

        
        self.fire_x += self.fire_vx * self.dt
        self.fire_y += self.fire_vy * self.dt
        self.water_x += self.water_vx * self.dt
        self.water_y += self.water_vy * self.dt

        
        def clamp_char(x, y, vx, vy):
            
            vx *= 0.9
            if abs(vx) < 1.0:  
                vx = 0.0

            
            x = max(0.0, min(self.W, x))

            
            if y >= self.ground_y:
                y = self.ground_y
                vy = 0.0
                on_ground = True
            else:
                on_ground = False

            
            if y < 0:
                y = 0.0
                vy = 0.0

            return x, y, vx, vy, on_ground

        (self.fire_x, self.fire_y, self.fire_vx, self.fire_vy, self.fire_on_ground) = \
            clamp_char(self.fire_x, self.fire_y, self.fire_vx, self.fire_vy)

        (self.water_x, self.water_y, self.water_vx, self.water_vy, self.water_on_ground) = \
            clamp_char(self.water_x, self.water_y, self.water_vx, self.water_vy)

        
        self._update_goal_states()

    def _get_obs(self):
        
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

        
        lava_x1, _, lava_x2, _ = self.lava_rect
        water_x1, _, water_x2, _ = self.water_rect
        lava_center_x = 0.5 * (lava_x1 + lava_x2)
        water_center_x = 0.5 * (water_x1 + water_x2)
        lava_center_x_norm = (lava_center_x / self.W) * 2.0 - 1.0
        water_center_x_norm = (water_center_x / self.W) * 2.0 - 1.0

        same_side = 1.0 if (self.fire_x < self.W / 2) == (self.water_x < self.W / 2) else 0.0

        time_remaining = 1.0 - (self.steps / self.max_steps)
        time_remaining = float(np.clip(time_remaining, 0.0, 1.0))

        obs = np.array(
            [
                fire_x_norm,           
                fire_y_norm,           
                fire_vx_norm,          
                fire_vy_norm,          
                float(self.fire_on_ground),  
                float(self.water_on_ground), 
                fire_goal_x_norm,      
                water_goal_x_norm,     
                float(np.clip(fire_to_goal, 0.0, 1.0)),   
                float(np.clip(water_to_goal, 0.0, 1.0)),  
                float(np.clip(min_goal_dist, 0.0, 1.0)),  
                same_side,             
                time_remaining,        
                lava_center_x_norm,    
                water_center_x_norm,   
            ],
            dtype=np.float32,
        )

        return obs

    
    
    
    def _ensure_render(self):
        if pygame is None:
            raise RuntimeError("pygame must be installed for render_mode='human'")

        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Fire & Water RL Env")
            self.screen = pygame.display.set_mode((self.W, self.H))
            self.clock = pygame.time.Clock()

    def _render_frame(self, return_array=False):
        if pygame is None:
            return

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((0, 0, 0))

        
        pygame.draw.rect(
            self.screen,
            (80, 80, 80),
            (0, self.ground_y, self.W, self.H - self.ground_y),
        )

        
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

        
        door_w = 30
        door_h = 60
        
        pygame.draw.rect(
            self.screen,
            (255, 80, 80),
            (self.fire_goal_x - door_w / 2, self.fire_goal_y - door_h, door_w, door_h),
        )
        
        pygame.draw.rect(
            self.screen,
            (80, 160, 255),
            (self.water_goal_x - door_w / 2, self.water_goal_y - door_h, door_w, door_h),
        )

        
        radius = getattr(self, "char_radius", 12)
        pygame.draw.circle(
            self.screen, (255, 50, 50), (int(self.fire_x), int(self.fire_y - radius)), radius
        )
        pygame.draw.circle(
            self.screen, (50, 150, 255), (int(self.water_x), int(self.water_y - radius)), radius
        )

        pygame.display.flip()

        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

        if return_array:
            
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )