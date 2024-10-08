import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# Set up device for GPU computations
DEVICE = torch.device("mps") if torch.has_mps else torch.device("cpu")

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()

        self.fc1 = nn.Linear(input_shape[0], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_space)

        self.log_std = nn.Parameter(torch.ones(1, action_space) * 0.01)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))
        log_std = torch.clamp(self.log_std, -20, 2)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist

class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.fc1 = nn.Linear(input_shape[0], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOMemory:
    def __init__(self, batch_size, max_memory_size=1000):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def shuffle_and_crop_memory(self):
        # Shuffle the data
        combined = list(zip(self.states, self.actions, self.probs, self.vals, self.rewards, self.dones))
        np.random.shuffle(combined)
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = zip(*combined)
        
        # Convert to lists again after shuffling
        self.states = list(self.states)
        self.actions = list(self.actions)
        self.probs = list(self.probs)
        self.vals = list(self.vals)
        self.rewards = list(self.rewards)
        self.dones = list(self.dones)

        # Crop the memory if it exceeds the max size
        if len(self.states) > self.max_memory_size:
            self.states = self.states[:self.max_memory_size]
            self.actions = self.actions[:self.max_memory_size]
            self.probs = self.probs[:self.max_memory_size]
            self.vals = self.vals[:self.max_memory_size]
            self.rewards = self.rewards[:self.max_memory_size]
            self.dones = self.dones[:self.max_memory_size]

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPOAgent:
    def __init__(self, n_actions, input_dims, max_memory_size=40000):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 5
        self.gae_lambda = 0.9

        self.actor = ActorNetwork(input_dims, n_actions).to("cpu")  # Keep actor on CPU
        self.critic = CriticNetwork(input_dims).to(DEVICE)  # Critic on GPU
        self.memory = PPOMemory(batch_size=256, max_memory_size=max_memory_size)

    def choose_action(self, observation):
        # Action selection on CPU
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to("cpu")
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(state.to(DEVICE))

        return action.numpy(), log_prob.detach().numpy(), value.item()

    def learn(self):
        print(f"Collected {len(self.memory.states)} data entries.")

        self.memory.shuffle_and_crop_memory()

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

        vals_arr = torch.tensor(vals_arr, dtype=torch.float32).to(DEVICE)
        reward_arr = torch.tensor(reward_arr, dtype=torch.float32).to(DEVICE)
        dones_arr = torch.tensor(dones_arr, dtype=torch.float32).to(DEVICE)
        advantage = torch.zeros_like(reward_arr)

        for t in range(len(reward_arr) - 1):
            delta = reward_arr[t:-1] + self.gamma * vals_arr[t + 1:] * (1 - dones_arr[t:-1]) - vals_arr[t:-1]
            advantage[t:-1] += delta

        for batch in batches:
            states = torch.tensor(state_arr[batch], dtype=torch.float32).to(DEVICE)
            old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(DEVICE)
            actions = torch.tensor(action_arr[batch], dtype=torch.float32).to("cpu")  # Actions kept on CPU

            dist = self.actor(states.to("cpu"))
            critic_value = self.critic(states).squeeze()

            new_probs = dist.log_prob(actions).sum(axis=-1).to(DEVICE)

            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + vals_arr[batch]
            critic_loss = F.mse_loss(critic_value, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()

        return actor_loss.item(), critic_loss.item()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, directory="./"):
        torch.save(self.actor.state_dict(), os.path.join(directory, "PPO_Actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "PPO_Critic.pth"))

    def load_models(self, directory="./"):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "PPO_Actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "PPO_Critic.pth")))

