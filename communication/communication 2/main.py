import mujoco_renderer
import numpy as np
import time
from asset_components import create_ant_model

def main():
    try:
        num_creatures = 9  # Number of creatures
        action_size = 12   # Action size per creature (adjust based on your model's requirement)
        max_steps = 100   # Max steps per episode
        episodes = 15      # Number of episodes
        flag_starting_radius = 3.5

        base_path = "/Users/andrewgordienko/Documents/body engine/communication/communication 2"

        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")

            # Generate a new ant model and save it to a file
            xml_string, leg_info = create_ant_model(flag_starting_radius)
            xml_filename = f'{base_path}/xml_world_episode_{episode}.xml'
            
            with open(xml_filename, 'w') as file:
                file.write(xml_string)
            
            print("Initializing environment...")
            env = mujoco_renderer.CustomAntEnv(xml_filename, max_steps)
            print("Environment initialized.")

            for step in range(max_steps):
                if env.should_close():
                    break

                reward = env.calculateReward()
                observation = env.getObservation()
                done = env.isDone()
                print(f"Step: {step}, Reward: {reward}, Done: {done}")

                # Generate a random action for each creature
                action = np.random.uniform(-1.0, 1.0, (num_creatures, action_size))
                
                # Send action to the environment
                env.setAction(action)

                # Render the environment
                env.render()

                if done:
                    break

            print(f"Episode {episode + 1} completed")

            # Explicitly delete the environment to free resources
            del env

    except KeyboardInterrupt:
        print("Environment shutdown initiated by user.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
