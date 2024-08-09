import mujoco_renderer
import numpy as np
import os
from ppo_agent import PPOAgent
from asset_components import create_ant_model

def main():
    num_creatures = 9  # Number of creatures
    obs_per_creature = 41  # Assuming each creature has 41 features (adjust based on your environment)
    action_size = 12   # Action size per creature
    max_steps = 1000  # Max steps per episode
    episodes = 15      # Number of episodes
    flag_starting_radius = 3.5

    base_path = "/Users/andrewgordienko/Documents/body engine/communication/communication 2"

    # Initialize the PPO agent
    input_dims = [obs_per_creature]  # Each creature has 41 features; do not flatten for the network input
    agent = PPOAgent(n_actions=action_size, input_dims=input_dims, max_memory_size=1000)

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

        done = False
        score = 0

        for step in range(max_steps):
            if env.should_close():
                break

            # Get observation and ensure it is processed correctly
            observation = env.getObservation()

            # Process each creature's observation individually
            actions = []
            log_probs = []
            values = []
            rewards = []

            for creature_idx in range(num_creatures):
                creature_obs = observation[creature_idx, :]  # Extract the observation for this creature
                action, log_prob, value = agent.choose_action(creature_obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # Convert actions to the appropriate format and set them in the environment
            combined_actions = np.vstack(actions)
            env.setAction(combined_actions)
            env.render()

            reward = env.calculateReward()
            done = env.isDone()

            for creature_idx in range(num_creatures):
                agent.remember(observation[creature_idx, :], actions[creature_idx], log_probs[creature_idx], values[creature_idx], reward[creature_idx], done)
            
            score += sum(reward)

            if done:
                break

        agent.learn()
        print(f'Episode {episode + 1} completed with score {score}')

        # Save the model after each episode
        agent.save_models(directory=base_path)

        # Explicitly delete the environment to free resources
        del env

if __name__ == "__main__":
    main()
