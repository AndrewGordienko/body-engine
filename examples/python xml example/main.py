from asset_components import create_ant_model
from environment_class import CustomAntEnv
from dm_control import viewer
import numpy as np

xml_string, leg_info = create_ant_model(3.5) # integer is radius
environment = CustomAntEnv(xml_string, leg_info)

# Save the XML string to a file
xml_filename = 'ant_model.xml'

with open(xml_filename, 'w') as file:
    file.write(xml_string)

print(f"XML file saved as {xml_filename}")

# Launch the viewer with the environment and policy
viewer.launch(environment, policy=environment.policy)