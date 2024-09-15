import jax
import jax.numpy as jnp
import mujoco
import time
from itertools import product
from creature_exploration.asset_components import create_ant_model
from creature_exploration.environment_class import CustomAntEnv  # Using your environment class

# Define possible values for legs (4 legs)
legs = range(4, 5)

# Define possible values for subparts per leg (2 to 3 subparts)
subparts_range = range(2, 4)

# Generate all possible combinations of legs with subparts
combinations = []
for num_legs in legs:
    subpart_combinations = product(subparts_range, repeat=num_legs)
    for subparts in subpart_combinations:
        combinations.append((num_legs, subparts))

# Function to generate random actions
def random_actions(rng_key, num_joints):
    return jax.random.uniform(rng_key, (num_joints,), minval=-1, maxval=1)

# Function to create environments for each creature
def create_creature_environments(num_creatures, xml_string, leg_info):
    environments = []
    for _ in range(num_creatures):
        env = CustomAntEnv(xml_string, leg_info)
        environments.append(env)
    return environments

# Preload all environments and return them as a list
def preload_and_prepare_environments(num_creatures):
    env_data = []

    for num_legs, subparts in combinations:
        # Print statement to mark start of XML processing
        print(f"Processing configuration: {num_legs} legs, subparts: {subparts}")
        config_start_time = time.time()

        # Generate the filename for the XML
        subparts_str = "_".join(map(str, subparts))
        xml_filename = f'ant_model_{num_legs}legs_{subparts_str}subparts.xml'

        # Generate the XML for the ant model using the create_ant_model function
        create_ant_model(num_legs, list(subparts), num_creatures=1, xml_filename=xml_filename)

        # Load the XML as a string
        with open(xml_filename, 'r') as file:
            xml_string = file.read()

        # Mark the start of environment creation
        print(f"Starting to create environments for {xml_filename}")
        env_creation_start = time.time()

        # Calculate the total number of joints
        leg_info = subparts

        # Create the environments for all creatures
        environments = create_creature_environments(num_creatures, xml_string, leg_info)

        env_creation_end = time.time()
        print(f"Finished creating environments for {xml_filename} in {env_creation_end - env_creation_start:.4f} seconds")

        # Store the environment data
        env_data.append({
            'xml_filename': xml_filename,
            'environments': environments,
        })

        config_end_time = time.time()
        print(f"Total time for configuration {xml_filename}: {config_end_time - config_start_time:.4f} seconds")

    return env_data

# Preload and prepare all environments
num_creatures = 1000  # Simulate 10 creatures per XML configuration
environments = preload_and_prepare_environments(num_creatures)

# Simulation parameters
timesteps_per_iteration = 25
iterations = 15

# Timing for parallel simulation
parallel_iteration_times = []  # List to store iteration times
env_step_times = []  # List to store times for each environment's steps
total_data_samples = 0  # To keep track of total data samples collected

# Run the simulation for all environments
for iteration in range(iterations):
    iteration_start_time = time.time()

    print(f"Iteration {iteration + 1}/{iterations} started for all environments.")

    # Iterate over each environment configuration and step each creature
    for env in environments:
        env_start_time = time.time()  # Start timer for this environment

        env_instances = env['environments']

        # Step through each environment in parallel for 25 timesteps and log each step time
        for step in range(timesteps_per_iteration):
            step_start_time = time.time()

            # Generate random actions for all creatures
            rng = jax.random.PRNGKey(step)
            actions = [random_actions(rng, sum(env_instance.leg_info)) for env_instance in env_instances]

            # Step each environment with its respective actions
            for i, custom_env in enumerate(env_instances):
                custom_env.step(actions[i])

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
