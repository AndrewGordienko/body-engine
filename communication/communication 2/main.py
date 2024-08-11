import mujoco_renderer
import numpy as np
import os
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from asset_components import create_ant_model

def main():
    num_creatures = 9
    obs_per_creature = 41
    action_size = 12
    max_steps = 1000
    episodes = 15
    flag_starting_radius = 3.5

    base_path = "/Users/andrewgordienko/Documents/body engine/communication/communication 2"

    input_dims = [obs_per_creature]
    agent = PPOAgent(n_actions=action_size, input_dims=input_dims, max_memory_size=1000)

    # Initialize plotting
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    fig.subplots_adjust(hspace=0.4)

    episode_scores = []
    actor_losses = []
    critic_losses = []

    # Initialize legend only once
    axs[0].plot([], [], label='Score')
    axs[1].plot([], [], label='Actor Loss', color='orange')
    axs[2].plot([], [], label='Critic Loss', color='green')

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

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

            observation = env.getObservation()
            actions = []
            log_probs = []
            values = []

            for creature_idx in range(num_creatures):
                creature_obs = observation[creature_idx, :]
                action, log_prob, value = agent.choose_action(creature_obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

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

        avg_actor_loss, avg_critic_loss = agent.learn()
        episode_scores.append(score)
        actor_losses.append(avg_actor_loss)
        critic_losses.append(avg_critic_loss)
        print(f'Episode {episode + 1} completed with score {score}')

        # Update plots
        axs[0].plot(episode_scores)
        axs[0].set_title('Score per Episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Score')

        axs[1].plot(actor_losses, color='orange')
        axs[1].set_title('Actor Loss per Episode')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Loss')

        axs[2].plot(critic_losses, color='green')
        axs[2].set_title('Critic Loss per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Loss')

        for ax in axs:
            ax.legend(loc='upper right')
            ax.grid(True)

        fig.canvas.draw()
        fig.canvas.flush_events()

        agent.save_models(directory=base_path)
        del env

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
