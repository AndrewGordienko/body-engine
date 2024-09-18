import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

class Environment:
    def extract_observations(self, batch_data, target_position):
        # Define the maximum number of joints (for padding)
        max_num_joints = 12

        # Vectorized extraction for all creatures in the batch
        
        # Extract torso positions (assuming the first body is the torso)
        torso_positions = batch_data.xpos[:, 0, :]  # Positions of all torsos in world coordinates
        
        # Extract torso velocities (assuming the first body is the torso)
        torso_velocities = batch_data.cvel[:, 0, :]  # Velocities of all torsos

        # Calculate distances to the target (vectorized across all creatures)
        distances_to_target = jnp.linalg.norm(torso_positions - target_position, axis=1)

        # Extract joint angles and forces (vectorized across all creatures)
        joint_angles = batch_data.qpos  # Joint positions (angles) for all creatures
        joint_forces = batch_data.qfrc_actuator  # Forces applied to the joints for all creatures

        # Calculate the padding for joint angles and forces
        pad_joint_angles = max(0, max_num_joints - joint_angles.shape[1])
        pad_joint_forces = max(0, max_num_joints - joint_forces.shape[1])

        # Pad the joint angles and forces to max_num_joints to make them consistent
        padded_joint_angles = jnp.pad(
            joint_angles, ((0, 0), (0, pad_joint_angles)),
            mode='constant', constant_values=0
        )

        padded_joint_forces = jnp.pad(
            joint_forces, ((0, 0), (0, pad_joint_forces)),
            mode='constant', constant_values=0
        )

        # Combine everything into a fixed-sized observation array
        observations = jnp.concatenate([
            torso_positions,                     # shape (num_creatures, 3)
            torso_velocities,                    # shape (num_creatures, 6)
            distances_to_target[:, None],        # shape (num_creatures, 1)
            padded_joint_angles,                 # shape (num_creatures, max_num_joints)
            padded_joint_forces                  # shape (num_creatures, max_num_joints)
        ], axis=1)                               # Final shape (num_creatures, n)
        
        return observations

    def calculate_rewards(self, batch_data, target_positions, step_counts, speed_reward_factor=1.0, energy_penalty_factor=0.00005):
        """
        Calculate rewards for all creatures in the batch.
        
        Args:
        - batch_data: The simulation data after the step.
        - target_positions: The positions of the targets (flags) for all creatures.
        - step_counts: The number of steps each creature has taken.
        - speed_reward_factor: Scaling factor for the speed reward.
        - energy_penalty_factor: Scaling factor for the energy penalty.

        Returns:
        - rewards: A 1D array of rewards for all creatures.
        """

        # Extract torso positions and joint controls from batch data
        torso_positions = batch_data.xpos[:, 0, :]  # Extract torso positions (assumed to be the first body)
        joint_controls = batch_data.ctrl            # Extract the control inputs (forces applied to joints)

        # Calculate distances to the target (vectorized)
        distances_to_target = jnp.linalg.norm(torso_positions - target_positions, axis=1)

        # Calculate speed reward (1 / (1 + step_count)) for each creature
        speed_rewards = speed_reward_factor / (1 + step_counts)

        # Calculate energy used as the sum of the absolute values of joint controls
        energy_used = jnp.sum(jnp.abs(joint_controls), axis=1)
        energy_penalties = energy_used * energy_penalty_factor

        # Calculate flag reached reward (reward = 10 if distance to target < 0.1, else 0)
        flag_reached_rewards = jnp.where(distances_to_target < 0.1, 10.0, 0.0)

        # Calculate the total reward for each creature
        total_rewards = speed_rewards + flag_reached_rewards - energy_penalties

        return total_rewards
