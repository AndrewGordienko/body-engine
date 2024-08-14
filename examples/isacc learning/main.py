import jax
import mujoco
from mujoco import mjx, Renderer
import jax.numpy as jp
import numpy as np
import cv2  # Required for displaying the simulation video

# Load humanoid.xml model
xml_path = "humanoid.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
renderer = Renderer(mj_model)  # Use Renderer for rendering

# Place the model and data on the GPU
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Print qpos types to confirm
print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# Enable joint visualization option
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Simulation parameters
duration = 3.8  # seconds
framerate = 60  # Hz

# Run the simulation with MuJoCo and render
frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

# Save or display the video using OpenCV
for frame in frames:
    cv2.imshow('Simulation', frame)
    cv2.waitKey(int(1000 / framerate))
cv2.destroyAllWindows()

# Run the simulation on GPU with MJX
jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

# Save or display the video using OpenCV
for frame in frames:
    cv2.imshow('Simulation', frame)
    cv2.waitKey(int(1000 / framerate))
cv2.destroyAllWindows()

# Run environments in parallel using MJX
rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 10)  # Number of humanoids

# Create a batch of environments
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (mjx_data.qpos.shape[0],))))(rng)

# Parallel step across all environments
jit_step_parallel = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step_parallel(mjx_model, batch)

# Display or inspect the results
print([data.qpos for data in batch])

# Copy batched mjx.Data back to MuJoCo for rendering
batched_mj_data = [mjx.get_data(mj_model, d) for d in batch]

# Render each humanoid's final state
final_frames = []
for data in batched_mj_data:
    renderer.update_scene(data, scene_option=scene_option)
    pixels = renderer.render()
    final_frames.append(pixels)

# Display all final frames
combined_frame = np.concatenate(final_frames, axis=1)
cv2.imshow('Final State', combined_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
