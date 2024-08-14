import mujoco
import jax
import numpy as np
from mujoco import mjx, Renderer
import cv2  # Required for displaying the simulation video

# Define the XML model
xml = """
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_string(xml)
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
rng = jax.random.split(rng, 4096)
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)

print(batch.qpos)

# Copy batched mjx.Data back to MuJoCo
batched_mj_data = mjx.get_data(mj_model, batch)
print([d.qpos for d in batched_mj_data])
