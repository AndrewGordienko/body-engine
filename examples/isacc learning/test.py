import jax
import mujoco
from mujoco import mjx, Renderer
import jax.numpy as jp
import cv2  # Required for displaying the simulation video

# Define the base XML for the spheres
sphere_xml = """
<mujoco model="spheres_simulation">
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    {spheres}
  </worldbody>
</mujoco>
"""

# Template for each sphere
sphere_template = """
<body name="sphere_{index}" pos="{x_pos} {y_pos} 0.5">
  <freejoint/>
  <geom name="sphere_{index}" type="sphere" size="0.2" density="0.1"/>
</body>
"""

# Generate XML for the spheres
num_spheres = 100
spacing = 2.0  # Spacing between spheres
spheres_xml = "\n".join([
    sphere_template.format(index=i, x_pos=(i % 5) * spacing, y_pos=(i // 5) * spacing)
    for i in range(num_spheres)
])
full_sphere_xml = sphere_xml.format(spheres=spheres_xml)

# Create MuJoCo model, data, and renderer
mj_model = mujoco.MjModel.from_xml_string(full_sphere_xml)
mj_data = mujoco.MjData(mj_model)
renderer = Renderer(mj_model)

# Place the model and data on the GPU using MJX
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Simulation parameters
duration = 1  # seconds
framerate = 60  # Hz

# Define a random action generator for the spheres
def select_random_forces(rng, num_dofs):
    return jax.random.uniform(rng, shape=(num_dofs,), minval=-0.5, maxval=0.5)

# Initialize random key for JAX
rng = jax.random.PRNGKey(0)

# JIT compile the step function with vectorized forces
@jax.jit
def simulate_step(model, data, forces):
    data = data.replace(qfrc_applied=forces)
    return mjx.step(model, data)

# Precompute the number of steps
num_steps = int(duration * framerate)

# Pre-generate random forces for all steps to avoid RNG overhead in the loop
rng, subkey = jax.random.split(rng)
forces = jax.random.uniform(subkey, shape=(num_steps, mj_model.nv), minval=-0.5, maxval=0.5)

# Run the simulation and render
frames = []

for i in range(num_steps):
    print(i)
    # Apply precomputed forces for the current step
    mjx_data = simulate_step(mjx_model, mjx_data, forces[i])
    
    # Render the scene
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data)
    pixels = renderer.render()
    frames.append(pixels)

# Display the frames as a video using OpenCV
for frame in frames:
    cv2.imshow('Simulation', frame)
    if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
