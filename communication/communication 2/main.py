import mujoco_renderer
import numpy as np
import time

def main():
    try:
        model_path = "/Users/andrewgordienko/Documents/body engine/communication/communication 2/ant_model.xml"
        print("Initializing environment...")
        env = mujoco_renderer.CustomAntEnv(model_path, 1000)
        print("Environment initialized.")

        num_creatures = 9  # Number of creatures
        action_size = 12   # Action size per creature (adjust based on your model's requirement)

        while not env.should_close():
            reward = env.calculateReward()
            print("Reward:", reward)
            observation = env.getObservation()
            done = env.isDone()
            print("Done:", done)

            # Generate a random action for each creature
            action = np.random.uniform(-1.0, 1.0, (num_creatures, action_size))
            
            # Send action to the environment
            env.setAction(action)

            # Render the environment
            env.render()

            # Optional: Adjust the sleep time as needed
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("Environment shutdown initiated by user.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
