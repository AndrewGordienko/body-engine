import numpy as np
import xml.etree.ElementTree as ET
from dm_control import mujoco, viewer
from dm_env import specs, TimeStep, StepType

class CustomAntEnv:
    def __init__(self, xml_string, leg_info, max_steps=2500, num_creatures=9):
        self.xml_string = xml_string
        self.leg_info = leg_info
        self.max_steps = max_steps
        self.step_count = 0
        self.num_creatures = num_creatures

        # Initialize previous_velocities later
        self.previous_velocities = None

        # Now call reset
        self.reset()

    def reset(self):
        self.physics = mujoco.Physics.from_xml_string(self.xml_string)

        if self.previous_velocities is None:
            MAX_LEGS = 4
            SUBPARTS_PER_LEG = 3
            self.previous_velocities = np.zeros(MAX_LEGS * SUBPARTS_PER_LEG * self.physics.model.nv // self.num_creatures)
            
        self.step_count = 0
        self.flag_positions = [self._get_flag_position(i) for i in range(self.num_creatures)]
        return TimeStep(StepType.FIRST, reward=0.0, discount=1.0, observation=self._get_observation())

    def step(self, action):
        # Flatten the action if it's a 2D array (for multiple creatures)
        if len(action.shape) > 1:
            action = action.flatten()

        # Process and set control for each creature
        for creature_id in range(self.num_creatures):
            # Calculate the start and end indices for this creature's actions
            start_idx = creature_id * self.physics.model.nu // self.num_creatures
            end_idx = (creature_id + 1) * self.physics.model.nu // self.num_creatures
            creature_action = action[start_idx:end_idx]
            
            # Set control for this creature's motors
            self.physics.data.ctrl[start_idx:end_idx] = creature_action

        # Advance the simulation
        self.physics.step()
        self.step_count += 1

        # Calculate rewards and observation
        rewards = [self._calculate_reward(i) for i in range(self.num_creatures)]
        observation = self._get_observation()
        done = self._is_done()
        step_type = StepType.LAST if done else StepType.MID

        # Sum the rewards if they are per creature, or use the total reward
        total_reward = np.sum(rewards) if isinstance(rewards, list) else rewards

        return TimeStep(step_type, rewards, discount=1.0, observation=observation)

    def _process_action_for_creature(self, action, creature_id):
        num_limbs = self._get_num_limbs_for_creature(creature_id)
        creature_specific_actions = action[creature_id * 12 : (creature_id + 1) * 12]  # Extract 12 actions for this creature

        # Initialize an array to store processed actions for this creature's motors
        creature_action = np.zeros(self.physics.model.nu // self.num_creatures)

        action_idx = 0  # Index for actions in creature_specific_actions
        motor_idx = 0  # Index to map to the correct motors
        for limb_id in range(num_limbs):
            num_subparts = self.leg_info[creature_id][limb_id]

            for subpart_id in range(num_subparts):
                start_idx = action_idx * 3
                end_idx = start_idx + 3

                if start_idx < len(creature_specific_actions):
                    subpart_action = creature_specific_actions[start_idx:end_idx]

                    # Assuming each subpart corresponds to one motor in sequence
                    creature_action[motor_idx : motor_idx + 3] = subpart_action
                    motor_idx += 3  # Increment to the next set of motors

                action_idx += 1  # Move to the next set of actions

                if action_idx >= 4:  # Limit to 4 limbs (12 actions / 3 actions per limb)
                    break

            if action_idx >= 4:  # Exit outer loop if all actions have been assigned
                break

        return creature_action

    def _get_num_limbs_for_creature(self, creature_id):
        return len(self.leg_info[creature_id])

    def _calculate_reward(self, creature_id):
        # Distance to Flag
        distance = np.linalg.norm(self._get_torso_position(creature_id) - self.flag_positions[creature_id])

        # Speed Reward - Higher reward for reaching the destination faster
        # You may adjust 'speed_reward_factor' to scale the reward
        speed_reward_factor = 1.0
        speed_reward = speed_reward_factor / (1 + self.step_count)

        # Energy Efficiency Reward - Penalize for more energy use
        energy_used = np.sum(np.abs(self.physics.data.ctrl[self.physics.model.nu * creature_id : self.physics.model.nu * (creature_id + 1)]))
        energy_penalty = energy_used * 0.00005  # Adjust the scaling factor as needed

        # High Reward for Reaching the Flag
        if distance < 0.1:
            flag_reached_reward = 10
        else:
            flag_reached_reward = 0

        # Total Reward Calculation
        total_reward = speed_reward + flag_reached_reward - energy_penalty
        return total_reward


    def _get_upright_orientation(self, creature_id):
        # Calculate the orientation of the torso with respect to the vertical axis
        # This is a simplified example; you may need to adjust it based on your model
        z_axis = self.physics.named.data.geom_xmat[f'torso_geom_torso_{creature_id}'][2::3]
        vertical_alignment = np.dot(z_axis, np.array([0, 0, 1]))  # Dot product with vertical axis
        return vertical_alignment


    def _get_observation(self):
        # Constants for observation sizes
        MAX_LEGS = 4
        SUBPARTS_PER_LEG = 3
        DATA_POINTS_PER_SUBPART = 3
        DISTANCE_TO_TARGET_DIMS = 2  # Distance to target (x, y)

        # Initialize a list to store observations of all creatures
        all_creature_observations = []

        # Iterate through each creature
        for creature_id in range(self.num_creatures):
            # Initialize observation arrays for this creature
            leg_observation = np.zeros(MAX_LEGS * SUBPARTS_PER_LEG * DATA_POINTS_PER_SUBPART)
            distance_to_target = np.zeros(DISTANCE_TO_TARGET_DIMS)

            # Calculate leg observations for this creature
            for i in range(MAX_LEGS):
                if i < len(self.leg_info[creature_id]):
                    for j in range(SUBPARTS_PER_LEG):
                        if j < self.leg_info[creature_id][i]:
                            physics_idx = i * SUBPARTS_PER_LEG + j
                            angle = self.physics.data.qpos[physics_idx]
                            velocity = self.physics.data.qvel[physics_idx]
                            acceleration = (velocity - self.previous_velocities[physics_idx]) / 0.02
                            self.previous_velocities[physics_idx] = velocity
                            obs_idx = i * (SUBPARTS_PER_LEG * DATA_POINTS_PER_SUBPART) + j * DATA_POINTS_PER_SUBPART
                            leg_observation[obs_idx:obs_idx+3] = [angle, velocity, acceleration]

            # Calculate distance to target for this creature
            distance_to_target = self._calculate_distance_to_target(creature_id)

            # Combine observations for this creature into a single array
            creature_observation = np.concatenate([leg_observation, distance_to_target])

            # Add this creature's observation to the list
            all_creature_observations.append(creature_observation)

        # Concatenate all creatures' observations into one array
        all_observations = np.concatenate(all_creature_observations)
        return all_observations


    def _calculate_distance_to_target(self, creature_id):
        # Calculate distance to the target stick for the specified creature
        torso_position = self._get_torso_position(creature_id)
        flag_position = self._get_flag_position(creature_id)
        distance = flag_position[:2] - torso_position[:2]  # Only consider x and y coordinates
        return distance
    
    def _is_done(self):
        if self.step_count >= self.max_steps:
            return True
        for creature_id in range(self.num_creatures):
            distance = np.linalg.norm(self._get_torso_position(creature_id) - self.flag_positions[creature_id])
            if distance < 0.1:
                return True
        return False

    def _get_torso_position(self, creature_id):
        return self.physics.named.data.geom_xpos[f'torso_geom_torso_{creature_id}']

    def _get_flag_position(self, flag_id):
        return self.physics.named.data.geom_xpos[f'flag_{flag_id}']

    def action_spec(self):
        action_shape = (self.num_creatures, 12)  # Define the shape of the action array
        minimum = -1  # Define the minimum value of each action
        maximum = 1   # Define the maximum value of each action
        return specs.BoundedArray(shape=action_shape, dtype=np.float32, minimum=minimum, maximum=maximum)

    def observation_spec(self):
        observation_shape = (342,)  # 36 data points per creature
        return specs.Array(shape=observation_shape, dtype=np.float32)

    def policy(self, time_step):
        # You need to define a policy function that takes the current time_step and returns actions
        # For simplicity, this example returns zero actions, which you might want to modify
        return np.zeros(self.action_spec().shape)
