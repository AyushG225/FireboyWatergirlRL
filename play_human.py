# play_human.py
import sys
import argparse
import pygame

from firewater_env import FireWaterEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Play Fire & Water manually")
    parser.add_argument(
        "--level",
        type=int,
        choices=range(5),
        default=0,
        help="Starting level id (0–4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create env in human render mode
    env = FireWaterEnv(render_mode="human", level_id=args.level)
    obs, info = env.reset()

    pygame.init()

    print("Controls:")
    print("  Fire (red):   LEFT / RIGHT arrows to move, UP arrow to jump")
    print("  Water (blue): A / D to move, W to jump")
    print("  R: reset current level")
    print("  N: next level")
    print("  P: previous level")
    print("  ESC / Q: quit")
    print(f"Starting on level {env.level_id} (0-based)")

    clock = pygame.time.Clock()
    running = True

    # To avoid repeating level switches on a held key, track key state edges
    prev_keys = pygame.key.get_pressed()

    while running:
        # Pump events so window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # ---------- Global controls ----------
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            running = False

        # Edge detection for R / N / P (only trigger on key-down)
        def just_pressed(k):
            return keys[k] and not prev_keys[k]

        if just_pressed(pygame.K_r):
            print("Resetting level...")
            obs, info = env.reset()

        if just_pressed(pygame.K_n):
            env.level_id = (env.level_id + 1) % len(env.levels)
            print(f"Switching to level {env.level_id}")
            obs, info = env.reset()

        if just_pressed(pygame.K_p):
            env.level_id = (env.level_id - 1) % len(env.levels)
            print(f"Switching to level {env.level_id}")
            obs, info = env.reset()

        # ---------- Build commands for manual_step ----------
        # Fire controls: arrows
        fire_cmd = None
        fire_left = keys[pygame.K_LEFT]
        fire_right = keys[pygame.K_RIGHT]
        fire_jump = keys[pygame.K_UP]

        if fire_left and not fire_right:
            fire_cmd = "left"
        elif fire_right and not fire_left:
            fire_cmd = "right"

        # If jump is held, combine it with movement if any
        if fire_jump:
            if fire_cmd == "left":
                fire_cmd = "left_jump"
            elif fire_cmd == "right":
                fire_cmd = "right_jump"
            else:
                fire_cmd = "jump"

        # Water controls: WASD
        water_cmd = None
        water_left = keys[pygame.K_a]
        water_right = keys[pygame.K_d]
        water_jump = keys[pygame.K_w]

        if water_left and not water_right:
            water_cmd = "left"
        elif water_right and not water_left:
            water_cmd = "right"

        if water_jump:
            if water_cmd == "left":
                water_cmd = "left_jump"
            elif water_cmd == "right":
                water_cmd = "right_jump"
            else:
                water_cmd = "jump"

        # Step the environment in manual mode
        env.manual_step(fire_cmd, water_cmd)

        # Check game state: success / death / timeout
        if env._fell_into_wrong_pool():
            print("💀 Fell into the wrong pool! Resetting level...")
            obs, info = env.reset()

        elif env._both_at_goals():
            print("🎉 Level complete! Resetting level...")
            obs, info = env.reset()

        elif env.steps >= env.max_steps:
            print("⏰ Timeout! Resetting level...")
            obs, info = env.reset()

        # Render one frame
        env.render()

        # Store previous key state for edge detection
        prev_keys = keys

        # Cap to env FPS
        clock.tick(env.metadata.get("render_fps", 30))

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
