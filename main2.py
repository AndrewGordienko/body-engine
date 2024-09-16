import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import time
from itertools import product
from creature_exploration.asset_components import create_ant_model

# Define possible values for legs (1 to 4 legs)
legs = range(4, 5)

# Define possible values for subparts per leg (2 to 3 subparts)
subparts_range = range(2, 4)

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

# Preload all environments and return them as a list
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

# Preload and prepare all environments
num_creatures = 4000  # Simulate 10 creatures per XML configuration
environments = preload_and_prepare_environments(num_creatures)

# Simulation parameters
timesteps_per_iteration = 25
iterations = 15

# Timing for parallel simulation
parallel_iteration_times = []  # List to store iteration times
env_step_times = []  # List to store times for each environment's steps
total_data_samples = 0  # To keep track of total data samples collected
iteration_times = []

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

        # Load model and data onto GPU just before stepping
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.put_data(mj_model, mj_data)

        # Create a batch of creatures
        rng_batch = jax.random.split(rng, num_creatures)
        batch = jax.vmap(lambda rng: mjx_data)(rng_batch)

        # JIT compile and pre-warm parallel stepping function
        start_parallel_load_time = time.time()
        dummy_actions = random_actions(rng, num_creatures, num_joints)
        _ = jit_step_parallel(mjx_model, batch, dummy_actions)  # Pre-warm
        end_parallel_load_time = time.time()

        print(f"Time to warm-up JIT and load batch for {num_creatures} creatures in ({env['xml_filename']}): {end_parallel_load_time - start_parallel_load_time:.4f} seconds")

        # Step through the environment in parallel for 25 timesteps and log each step time
        for step in range(timesteps_per_iteration):
            step_start_time = time.time()

            actions = random_actions(rng, num_creatures, num_joints)
            batch = jit_step_parallel(mjx_model, batch, actions)  # Step the environment

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"Step {step + 1}/{timesteps_per_iteration} for {env['xml_filename']} took {step_duration:.4f} seconds.")

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

# Print individual environment step times
# print("\nEnvironment Step Times:")
# for env_filename, step_time in env_step_times:
#    print(f"{env_filename}: {step_time:.4f} seconds")
