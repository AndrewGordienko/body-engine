mport jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

class CustomAntEnv:
    def __init__(self, xml_string, leg_info, max_steps=2500, num_creatures=100):
        self.xml_string = xml_string
        self.leg_info = leg_info
        self.max_steps = max_steps
        self.num_creatures = num_creatures
        self.step_count = 0

        # Initialize velocities and other states
        self.previous_velocities = jax.device_put(jnp.zeros(self.num_creatures))

        # Load the MuJoCo model and data onto the GPU using MJX
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Move the model and data to the GPU
        self.mj_model_gpu = mjx.put_model(self.mj_model)
        self.mj_data_gpu = mjx.put_data(self.mj_model, self.mj_data)

        # Initialize PRNG
        self.rng_key = jax.random.PRNGKey(0)

    def reset(self):
        self.step_count = 0
        self.previous_velocities = jax.device_put(jnp.zeros(self.num_creatures))
        return self._get_observations()

    @jax.jit
    def step(self, actions):
        # Set the actions for all creatures and step the environment
        self.mj_data_gpu = self.mj_data_gpu.replace(ctrl=actions)
        self.mj_data_gpu = mjx.step(self.mj_model_gpu, self.mj_data_gpu)
        self.step_count += 1

        # Get new observations and compute reward
        observations = self._get_observations()
        reward = self._calculate_reward()
        done = self.step_count >= self.max_steps

        return observations, reward, done

    @jax.jit
    def _get_observations(self):
        observations = []
        for creature_id in range(self.num_creatures):
            torso_position = self._get_torso_position(creature_id)
            torso_velocity = self._get_torso_velocity(creature_id)
            distance_to_target = self._calculate_distance_to_target(creature_id)
            observations.append(jnp.concatenate([torso_position, [torso_velocity], [distance_to_target]]))
        return jax.device_put(jnp.array(observations))

    def _calculate_reward(self):
        rewards = []
        for creature_id in range(self.num_creatures):
            current_velocity = self._get_torso_velocity(creature_id)
            reward = current_velocity - self.previous_velocities[creature_id]
            rewards.append(reward)
            self.previous_velocities = self.previous_velocities.at[creature_id].set(current_velocity)
        return jnp.sum(jnp.array(rewards))

    def _get_torso_position(self, creature_id):
        torso_geom_id = self.mj_model.name2id(f'torso_{creature_id}_geom', 'geom')
        return self.mj_data_gpu.geom_xpos[torso_geom_id * 3:torso_geom_id * 3 + 3]

    def _get_torso_velocity(self, creature_id):
        joint_id = self.mj_model.name2id(f'torso_{creature_id}_root', 'joint')
        qvel_start = self.mj_model.jnt_dofadr[joint_id]
        return self.mj_data_gpu.qvel[qvel_start]

    def _calculate_distance_to_target(self, creature_id):
        torso_position = self._get_torso_position(creature_id)
        flag_position = self._get_flag_position(creature_id)
        return jnp.linalg.norm(torso_position[:2] - flag_position[:2])

    def _get_flag_position(self, creature_id):
        flag_geom_id = self.mj_model.name2id(f'flag_{creature_id}', 'geom')
        return self.mj_data_gpu.geom_xpos[flag_geom_id * 3:flag_geom_id * 3 + 3]
