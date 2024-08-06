import mujoco_renderer
import time

def main():
    try:
        model_path = "/Users/andrewgordienko/Documents/body engine/cpp xml example 2/ant_model.xml"
        print("Initializing environment...")
        env = mujoco_renderer.CustomAntEnv(model_path)
        print("Environment initialized.")

        # Example action vector
        action = [0.0] * 10  # Set the action size to 10

        while not env.should_close():
            reward = env.getReward()
            observation = env.getObservation(0)  # Pass the integer value 0 to C++
            print("Reward:", reward)
            print("Observation:", observation)

            # Send action to the environment
            env.setAction(action)

            # Render the environment
            env.render()

            time.sleep(0.01)  # Adjust the sleep time as needed

    except KeyboardInterrupt:
        print("Environment shutdown initiated by user.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
