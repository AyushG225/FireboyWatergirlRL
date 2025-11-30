import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from firewater_env import FireWaterEnv


def make_env():
    return FireWaterEnv(render_mode=None)


if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./tb_firewater",
        n_steps=4096,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_firewater")
