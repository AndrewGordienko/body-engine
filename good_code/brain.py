import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import jax.numpy as jnp

INPUT_SHAPE = 12
FC1_DIMS = 1024
FC2_DIMS = 512
ACTION_SPACE = 12 # 4 legs with 3 subparts
INPUT_SHAPE = 87
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda")

class actor_network(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()

        self.std = 0.5

        self.fc1 = nn.Linear(INPUT_SHAPE, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, action_space)

        self.log_std = nn.Parameter(torch.ones(1, action_space) * 0.01)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


    def net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Use tanh for the last layer

        return x
    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist

class critic_network(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Unpack the input_shape tuple when passing it to nn.Linear
        self.fc1 = nn.Linear(INPUT_SHAPE, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, mem_size, observation_dim, action_dim, device):
        self.mem_count = 0
        self.mem_size = mem_size
        self.device = device

        # Initialize buffer for states, actions, rewards, probs, and vals
        self.states = np.zeros((mem_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((mem_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.probs = np.zeros(mem_size, dtype=np.float32)
        self.vals = np.zeros(mem_size, dtype=np.float32)

    def add(self, states, actions, rewards, log_probs, values):
        num_entries = states.shape[0]  # Number of entries (e.g., 4000 creatures)
        
        # Circular buffer logic: Determine where to start adding data
        mem_index = self.mem_count % self.mem_size
        end_index = mem_index + num_entries
        values = np.squeeze(values)

        if end_index <= self.mem_size:
            # No overflow: directly add to the buffer
            self.states[mem_index:end_index] = states
            self.actions[mem_index:end_index] = actions
            self.rewards[mem_index:end_index] = rewards
            self.probs[mem_index:end_index] = log_probs
            self.vals[mem_index:end_index] = values
        else:
            print("overflow")
            # Overflow: split into two parts
            first_part_size = self.mem_size - mem_index
            self.states[mem_index:] = states[:first_part_size]
            self.actions[mem_index:] = actions[:first_part_size]
            self.rewards[mem_index:] = rewards[:first_part_size]
            self.probs[mem_index:] = log_probs[:first_part_size]
            self.vals[mem_index:] = values[:first_part_size]

            remaining_part_size = num_entries - first_part_size
            self.states[:remaining_part_size] = states[first_part_size:]
            self.actions[:remaining_part_size] = actions[first_part_size:]
            self.rewards[:remaining_part_size] = rewards[first_part_size:]
            self.probs[:remaining_part_size] = log_probs[first_part_size:]
            self.vals[:remaining_part_size] = values[first_part_size:]

        # Increment memory count
        self.mem_count += num_entries

    def sample(self, batch_size):
        mem_max = min(self.mem_count, self.mem_size)
        batch_indices = np.random.choice(mem_max, batch_size, replace=False)

        # Sample data from the buffer
        states = torch.tensor(self.states[batch_indices], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions[batch_indices], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rewards[batch_indices], dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(self.probs[batch_indices], dtype=torch.float32).to(self.device)
        vals = torch.tensor(self.vals[batch_indices], dtype=torch.float32).to(self.device)

        return states, actions, rewards, log_probs, vals

    def clear(self):
        # Reset memory counter and clear all stored data
        self.mem_count = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.probs.fill(0)
        self.vals.fill(0)

class Agent:
    def __init__(self):
        self.actor = actor_network(INPUT_SHAPE, ACTION_SPACE).to(DEVICE)  # Ensure actor is on the correct device
        self.critic = critic_network(INPUT_SHAPE).to(DEVICE)  # Ensure critic is on the correct device
      
    def choose_action(self, observation):
        # Convert observation to a writable NumPy array if it's a JAX array (or NumPy array from JAX)
        if isinstance(observation, jnp.ndarray):
            observation = np.array(observation, copy=True)  # Copy ensures it's writable

        # Ensure observation is a 2D array for batch processing (if it's not already)
        if observation.ndim == 1:
            observation = np.expand_dims(observation, axis=0)

        # Convert observation to a tensor and move it to the correct device
        state = torch.tensor(observation, dtype=torch.float).to(DEVICE)
        
        # Forward pass through actor network to get the action distribution
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        
        # Forward pass through critic network to get the state values
        value = self.critic(state)  # The value is now a batch of values, one for each creature
        
        # Convert action and log_prob to NumPy arrays and move value to CPU
        action = action.cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        value = value.detach().cpu().numpy()  # Return the full batch of values (4000 in your case)

        return action, log_prob, value
