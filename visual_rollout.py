from time import sleep
from firewater_env import FireWaterEnv


def main():
    env = FireWaterEnv(render_mode="human")
    obs, info = env.reset()
    print("initial obs shape:", obs.shape)
    print("initial obs:", obs)

    for t in range(300):
        env.render()

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"t={t:03d}  action={action}  "
            f"reward={reward:.3f}  term={terminated}  trunc={truncated}"
        )

        sleep(1 / 30.0)

        if terminated or truncated:
            print("Episode finished.")
            break

    env.close()

    import matplotlib.pyplot as plt

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
