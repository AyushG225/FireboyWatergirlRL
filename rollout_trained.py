import time
from stable_baselines3 import PPO
from firewater_env import FireWaterEnv

if __name__ == "__main__":

    model = PPO.load("ppo_firewater")

    env = FireWaterEnv(render_mode="human")
    obs, info = env.reset()

    done = False
    truncated = False
    step_idx = 0

    while not (done or truncated):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)

        print(
            f"t={step_idx:03d} action={int(action)} reward={reward:+.3f} "
            f"done={done} trunc={truncated} info={info}"
        )

        env.render()
        time.sleep(1 / 30.0)

        step_idx += 1

    env.close()
