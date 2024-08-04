import mujoco_renderer
import time

def main():
    try:
        model_path = "/Users/andrewgordienko/Documents/body engine/cpp xml example 2/ant_model.xml"
        print("Initializing environment...")
        env = mujoco_renderer.CustomAntEnv(model_path)
        print("Environment initialized.")

        # Keep the environment running
        while not env.should_close():
            reward = env.get_reward()
            print("Reward:", reward)

            observation = env.get_observation()
            print("Observation:", observation)
            env.render()
            time.sleep(1.0 / 60.0)  # Aim for 60 FPS

    except KeyboardInterrupt:
        print("Environment shutdown initiated by user.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
