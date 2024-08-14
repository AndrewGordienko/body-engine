import numpy as np
import xml.etree.ElementTree as ET
import random

joint_ranges = {
    'hip': '-60 60',
    'knee': '-70 70',
    'ankle': '-30 30'
}

motor_gears = {
    'hip': 150,
    'knee': 150,
    'ankle': 150
}

joint_damping = {
    'hip': '10.0',
    'knee': '15.0',
    'ankle': '20.0'
}

class Torso:
    def __init__(self, name="torso", position=(0, 0, 0.75), size=None):
        self.name = name
        self.position = position
        self.size = size if size else (random.uniform(0.2, 0.5), random.uniform(0.1, 0.2), random.uniform(0.05, 0.1))

    def to_xml(self, layer, color):
        torso = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, self.position))})
        ET.SubElement(torso, 'geom', attrib={
            'name': f'torso_geom_{self.name}', 
            'type': 'box', 
            'size': ' '.join(map(str, self.size)), 
            'pos': '0 0 0', 
            'contype': '1', 
            'conaffinity': str(layer),
            'material': color
        })
        
        ET.SubElement(torso, 'joint', attrib={
            'name': f'{self.name}_root', 
            'type': 'free', 
            'armature': '0', 
            'damping': '0', 
            'limited': 'false'
        })
        ET.SubElement(torso, 'site', attrib={
            'name': f'{self.name}_site', 
            'pos': '0 0 0', 
            'type': 'sphere', 
            'size': '0.01'
        })

        return torso

class Leg:
    def __init__(self, name, torso_size, size):
        self.name = name
        self.torso_size = torso_size
        self.size = size
        self.subparts = 0

    def to_xml(self):
        edge_positions = [
            (0, self.torso_size[1]/2, 0),  # Right side
            (0, -self.torso_size[1]/2, 0),  # Left side
            (self.torso_size[0]/2, 0, 0),  # Front side
            (-self.torso_size[0]/2, 0, 0)  # Back side
        ]
        position = random.choice(edge_positions)

        leg = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, position))})

        upper_length = np.random.uniform(0.1, 0.2)
        lower_length = np.random.uniform(0.1, 0.2)
        foot_length = np.random.uniform(0.1, 0.2)

        upper_fromto = [0.0, 0.0, 0.0, upper_length, 0.0, 0.0]
        ET.SubElement(leg, 'geom', attrib={'name': self.name + '_upper_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, upper_fromto)), 'size': str(self.size)})
        ET.SubElement(leg, 'joint', attrib={'name': self.name + '_hip_joint', 'type': 'ball', 'damping': joint_damping['hip']})

        lower_fromto = [upper_length, 0.0, 0.0, upper_length + lower_length, 0.0, 0.0]
        lower_part = ET.SubElement(leg, 'body', attrib={'name': self.name + '_lower', 'pos': ' '.join(map(str, [upper_length, 0.0, 0.0]))})
        ET.SubElement(lower_part, 'geom', attrib={'name': self.name + '_lower_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, lower_fromto)), 'size': str(self.size)})

        ET.SubElement(lower_part, 'joint', attrib={'name': self.name + '_knee_joint', 'type': 'hinge', 'axis': '0 1 0', 'range': joint_ranges['knee'], 'damping': joint_damping['knee'], 'limited': 'true'})

        foot_fromto = [upper_length + lower_length, 0.0, 0.0, upper_length + lower_length + foot_length, 0.0, 0.0]
        foot_part = ET.SubElement(lower_part, 'body', attrib={'name': self.name + '_foot', 'pos': ' '.join(map(str, [upper_length + lower_length, 0.0, 0.0]))})
        ET.SubElement(foot_part, 'geom', attrib={'name': self.name + '_foot_geom', 'type': 'cylinder', 'fromto': ' '.join(map(str, foot_fromto)), 'size': str(self.size)})
        ET.SubElement(foot_part, 'joint', attrib={'name': self.name + '_ankle_joint', 'type': 'ball', 'damping': joint_damping['ankle']})

        self.subparts = 1  # upper part
        self.subparts += 1 if lower_length > 0 else 0
        self.subparts += 1 if foot_length > 0 else 0

        return leg, self.name + '_ankle_joint'