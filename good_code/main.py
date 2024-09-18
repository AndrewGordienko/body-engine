import jax
import jax.numpy as jnp
import mujoco                                                      
from mujoco import mjx
import time                                                                               
from itertools import product                      
from creature_exploration.asset_components import create_ant_model
import numpy as np                                                  
import torch                                   
import torch.nn as nn                                                 
import torch.nn.functional as F                                                     
import torch.optim as optim                                                              
from torch.autograd import Variable
from environment import Environment    
from brain import ReplayBuffer, Agent                      

# Simulation parameters
num_creatures = 4000  # Number of creatures across all configurations
timesteps_per_iteration = 25  # Number of timesteps per iteration
iterations = 15  # Number of iterations                                                    
                                                          
# Replay buffer settings                                                                       
MEM_SIZE = 1600000  # Capacity of 1.6 million transitions
BATCH_SIZE = 64  # Batch size for sampling                                
OBSERVATION_DIM = 87  # Number of observation values per creature
ACTION_DIM = 12  # Number of action values per creature             
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device (GPU if available)
                                                                                                         
# Initialize environment and replay buffer
custom_environment = Environment()                                 
replay_buffer = ReplayBuffer(MEM_SIZE, OBSERVATION_DIM, ACTION_DIM, DEVICE)
agent = Agent()
                         
# Example data for buffer insertion (dummy data)                                  
# observations = np.random.randn(num_creatures, OBSERVATION_DIM)  # Dummy observation data
# actions = np.random.randn(num_creatures, ACTION_DIM)  # Dummy action data
# rewards = np.random.randn(num_creatures)  # Dummy reward data   
# next_observations = np.random.randn(num_creatures, OBSERVATION_DIM)  # Dummy next state data
    
# Add the dummy data to the replay buffer
# replay_buffer.add(observations, actions, rewards, next_observations)                 
                                                     
# Optionally sample from the replay buffer (for learning purposes)
# sampled_states, sampled_actions, sampled_rewards, sampled_next_states = replay_buffer.sample(BATCH_SIZE)
                                                                                     
# Print message to confirm functionality
# print("Replay buffer setup and insertion complete.")                                  
                                                                              
# Timing variables for parallel simulation
parallel_iteration_times = []  # Store iteration times for parallel simulation
env_step_times = []  # Store the time taken for each environment step
total_data_samples = 0  # Track total data samples collected
iteration_times = []  # Store the times for each simulation iteration
 
# Define possible values for legs (1 to 4 legs)
legs = range(4, 5)                                                 
 
# Define possible values for subparts per leg (2 to 3 subparts)
subparts_range = range(3, 4)        
                                                
# Generate all possible combinations of legs with subparts
combinations = []
for num_legs in legs:
    subpart_combinations = product(subparts_range, repeat=num_legs)
    for subparts in subpart_combinations:
        combinations.append((num_legs, subparts))

# Function to step the simulation with random actions
def simulate_step(mjx_model, mjx_data, action):
    mjx_data = mjx_data.replace(ctrl=mjx_data.ctrl.at[:].set(action))
    return mjx.step(mjx_model, mjx_data)

# JIT compile the parallel step function
jit_step_parallel = jax.jit(jax.vmap(simulate_step, in_axes=(None, 0, 0)))

# Function to generate random actions
def random_actions(rng_key, num_creatures,num_joints):
    return jax.vmap(lambda key: jax.random.uniform(key, (num_joints,), minval=-1, maxval=1))(jax.random.split(rng_key, num_creatures))

# Function to preload all environments and return them as a list
def preload_and_prepare_environments(num_creatures):
    env_data = []

    for num_legs, subparts in combinations:
        # Generate the filename for the XML
        subparts_str = "_".join(map(str, subparts))
        xml_filename = f'ant_model_{num_legs}legs_{subparts_str}subparts.xml'

        # Generate the XML for the ant model using the create_ant_model function
        create_ant_model(num_legs, list(subparts), num_creatures=1, xml_filename=xml_filename)

        # Load the model from the XML
        mj_model = mujoco.MjModel.from_xml_path(xml_filename)
        mj_data = mujoco.MjData(mj_model)

        # Precompute number of joints and random number generators for each configuration
        num_joints = mj_data.ctrl.size
        rng = jax.random.PRNGKey(0)

        # Store the environment data without loading onto GPU yet
        env_data.append({
            'xml_filename': xml_filename,
            'mj_model': mj_model,
            'mj_data': mj_data,
            'num_joints': num_joints,
            'rng': rng
        })
    return env_data

environments = preload_and_prepare_environments(num_creatures)

target_position = jnp.array([0.0, 0.0, 0.0])  # Define target position for distance calculations

# Run the simulation for all environments
for iteration in range(iterations):
    iteration_start_time = time.time()

    print(f"Iteration {iteration + 1}/{iterations} started for all environments.")

    # Iterate over each environment and step it for 25 timesteps
    for env in environments:
        env_start_time = time.time()  # Start timer for this environment

        mj_model = env['mj_model']
        mj_data = env['mj_data']
        num_joints = env['num_joints']
        rng = env['rng']
        
        mujoco.mj_resetData(mj_model, mj_data)

        # Load model and data onto GPU just before stepping
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.put_data(mj_model, mj_data)

        # Create a batch of creatures
        rng_batch = jax.random.split(rng, num_creatures)
        batch = jax.vmap(lambda rng: mjx_data)(rng_batch)
        
        flag_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'flag_0')  # Get site ID for the flag
        target_position = mj_data.site_xpos[flag_site_id]  # Extract the 3D position of the flag

        # JIT compile and pre-warm parallel stepping function
        start_parallel_load_time = time.time()
        dummy_actions = random_actions(rng, num_creatures, num_joints)
        _ = jit_step_parallel(mjx_model, batch, dummy_actions)  # Pre-warm
        end_parallel_load_time = time.time()
        states = custom_environment.extract_observations(batch, target_position)

        print(f"Time to warm-up JIT and load batch for {num_creatures} creatures in ({env['xml_filename']}): {end_parallel_load_time - start_parallel_load_time:.4f} seconds")

        # Step through the environment in parallel for 25 timesteps and log each step time
        for step in range(timesteps_per_iteration):
            step_start_time = time.time()
            actions, log_probs, values = agent.choose_action(states)
            actions_ = jax.numpy.array(actions)
            # actions = random_actions(rng, num_creatures, num_joints)
            states = custom_environment.extract_observations(batch, target_position)
            batch = jit_step_parallel(mjx_model, batch, actions_)  # Step the environment

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"Step {step + 1}/{timesteps_per_iteration} for {env['xml_filename']} took {step_duration:.4f} seconds.")

            # Extract observations
            # next_states = custom_environment.extract_observations(batch, target_position)
            step_counts = jnp.full((num_creatures,), step)
            rewards = custom_environment.calculate_rewards(batch, target_position, step_counts)
            
            replay_buffer.add(states, actions, rewards, log_probs, values)

        env_end_time = time.time()  # End timer for this environment
        env_duration = env_end_time - env_start_time  # Time taken for this environment
        env_step_times.append((env['xml_filename'], env_duration))  # Store the time for this environment
        
        samples_this_body = num_creatures * timesteps_per_iteration

        print(f"Time to step through environment ({env['xml_filename']}) for {num_creatures} creatures: {env_duration:.4f} seconds")
       print(f"Samples collected through this configuration {samples_this_body}")

    iteration_end_time = time.time()
    iteration_duration = iteration_end_time - iteration_start_time
    parallel_iteration_times.append(iteration_duration)
    
    num_configs = len(environments)
    data_samples_this_iteration = num_configs * num_creatures * timesteps_per_iteration
    total_data_samples += data_samples_this_iteration

    print(f"Iteration {iteration + 1}/{iterations} completed in {iteration_duration:.4f} seconds.")
    print(f"Data samples collected in this iteration: {data_samples_this_iteration}")

total_samples = len(environments) * num_creatures * timesteps_per_iteration * iterations
print(f"\nTotal data samples collected after all iterations: {total_samples}")

# Print parallel iteration times and total time
print("\nParallel Iteration Times (in seconds):")
for i, time in enumerate(parallel_iteration_times, 1):
    print(f"Iteration {i}: {time:.4f} seconds")

total_time = sum(parallel_iteration_times)
print(f"\nTotal time for all iterations: {total_time:.4f} seconds")

# Optionally print individual environment step times
# print("\nEnvironment Step Times:")
# for env_filename, step_time in env_step_times:
#    print(f"{env_filename}: {step_time:.4f} seconds")
 
