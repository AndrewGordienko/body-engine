from asset_components import create_ant_model
from environment_class import CustomAntEnv
from dm_control import viewer
import numpy as np

xml_string, leg_info = create_ant_model(3.5)  # Adjust the parameter if needed
environment = CustomAntEnv(xml_string, leg_info)

# Launch the viewer with the environment and policy
viewer.launch(environment, policy=environment.policy)
