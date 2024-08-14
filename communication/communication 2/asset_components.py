import numpy as np
import xml.etree.ElementTree as ET
import random
from body_components import Torso, Leg
import math

joint_damping = {
    'hip': '5.0',
    'knee': '10.0',
    'ankle': '15.0'
}

motor_gears = {
    'hip': 200,
    'knee': 200,
    'ankle': 200
}

joint_ranges = {
    'hip': '-60 60',
    'knee': '-70 70',
    'ankle': '-30 30'
}

def create_assets_xml():
    assets = ET.Element('asset')

    ET.SubElement(assets, 'texture', attrib={
        'name': 'checkered',
        'type': '2d',
        'builtin': 'checker',
        'rgb1': '0.2 0.3 0.4',
        'rgb2': '0.9 0.9 0.9',
        'width': '512',
        'height': '512'
    })

    ET.SubElement(assets, 'material', attrib={
        'name': 'MatCheckered',
        'texture': 'checkered',
        'reflectance': '0.5'
    })

    ET.SubElement(assets, 'material', attrib={
        'name': 'MatPlane',
        'reflectance': '0.5',
        'shininess': '1',
        'specular': '1'
    })

    color_materials = {
        'red': '1 0 0 1', 
        'green': '0 1 0 1', 
        'blue': '0 0 1 1',
        'yellow': '1 1 0 1', 
        'purple': '0.5 0 0.5 1', 
        'orange': '1 0.5 0 1',
        'pink': '1 0.7 0.7 1', 
        'grey': '0.5 0.5 0.5 1', 
        'brown': '0.6 0.3 0 1',
        'cyan': '0 1 1 1',
        'magenta': '1 0 1 1',
        'lime': '0 1 0.5 1',
        'maroon': '0.5 0 0 1',
        'navy': '0 0 0.5 1',
        'olive': '0.5 0.5 0 1',
        'teal': '0 0.5 0.5 1',
        'aqua': '0 1 1 1',
        'fuchsia': '1 0 0.5 1',
        'salmon': '1 0.5 0.5 1',
        'coral': '1 0.5 0.3 1',
    }

    for name, rgba in color_materials.items():
        ET.SubElement(assets, 'material', attrib={'name': name, 'rgba': rgba})

    return assets

def create_floor_xml(size=(20, 20, 0.1)):
    return ET.Element('geom', attrib={'name': 'floor', 'type': 'plane', 'size': ' '.join(map(str, size)), 'pos': '0 0 0', 'material': 'MatCheckered'})

def create_flag_xml(flag_id, layer, color, center_position, dynamic_radius):
    angle = random.uniform(0, 2 * math.pi)
    flag_x = center_position[0] + dynamic_radius * math.cos(angle)
    flag_y = center_position[1] + dynamic_radius * math.sin(angle)
    flag_z = 0
    flag_position = (flag_x, flag_y, flag_z)
    flag_size = (0.05, 0.05, 0.5)
    return ET.Element('geom', attrib={
        'name': f'flag_{flag_id}', 
        'type': 'box', 
        'size': ' '.join(map(str, flag_size)), 
        'pos': ' '.join(map(str, flag_position)), 
        'material': color,
        'contype': '1', 
        'conaffinity': str(layer)
    })

def create_ant_model(flag_radius):
    num_creatures = 20
    
    mujoco_model = ET.Element('mujoco')
    ET.SubElement(mujoco_model, 'option', attrib={'timestep': '0.002'})

    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    world_size = (20, 20, 0.1)
    worldbody.append(create_floor_xml(size=world_size))

    actuator = ET.SubElement(mujoco_model, 'actuator')
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown', 
              'cyan', 'magenta', 'lime', 'maroon', 'navy', 'olive', 'teal', 'aqua', 'fuchsia',
              'salmon', 'coral']  

    creature_leg_info = {}
    grid_size = int(math.ceil(math.sqrt(num_creatures)))
    edge_offset = 9.5
    max_spread_positions = []

    for i in range(grid_size):
        for j in range(grid_size):
            if len(max_spread_positions) < num_creatures:
                x_pos = -edge_offset + 2 * edge_offset * i / (grid_size - 1)
                y_pos = -edge_offset + 2 * edge_offset * j / (grid_size - 1)
                max_spread_positions.append((x_pos, y_pos, 0.75))

    for creature_id in range(num_creatures):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]
        initial_position = max_spread_positions[creature_id]

        torso_obj = Torso(name=f'torso_{creature_id}', position=initial_position)
        torso_xml = torso_obj.to_xml(layer, color)
        worldbody.append(torso_xml)

        worldbody.append(create_flag_xml(creature_id, layer, color, initial_position, flag_radius))

        num_legs = random.randint(1, 4)
        leg_size = 0.04
        leg_info = []

        for i in range(num_legs):
            leg_name = f"leg_{creature_id}_{i+1}"

            leg_obj = Leg(leg_name, torso_obj.size, leg_size)
            leg_xml, foot_joint_name = leg_obj.to_xml()
            torso_xml.append(leg_xml)

            ET.SubElement(actuator, 'motor', attrib={
                'name': f'{leg_name}_hip_motor',
                'joint': f'{leg_name}_hip_joint',
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'gear': str(motor_gears['hip'])
            })

            if 'knee_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{leg_name}_knee_motor',
                    'joint': f'{leg_name}_knee_joint',
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['knee'])
                })

            if 'ankle_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{foot_joint_name}_motor',
                    'joint': foot_joint_name,
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['ankle'])
                })
            
            leg_info.append(leg_obj.subparts)
        
        creature_leg_info[creature_id] = leg_info

    sensors = ET.SubElement(mujoco_model, 'sensor')
    for creature_id in range(num_creatures):
        torso_name = f'torso_{creature_id}'
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{torso_name}_accel', 'site': f'{torso_name}_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{torso_name}_gyro', 'site': f'{torso_name}_site'})

    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, creature_leg_info