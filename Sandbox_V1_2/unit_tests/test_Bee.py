import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The BeeController class has these attributes:

        theta_noisemaker
        speed_noisemaker
        heading_noisemaker

        inputs

        theta_command
        speed_command
        heading_command
        theta_commands
        speed_commands
        heading_commands

        step_fun

        (plus what it inherits from Controller)

    The BeeControllerator class has these methods:

        __init__
        step

        (plus what it inherits from Controller)

'''

'''

    The Bee class has these attributes:

        sensors
        sensor_angles
        initial_sensor_angles

        max_speed
        theta_inertia
        speed_inertia

        controller

        heading
        headings

        speed
        speeds

        (plus what it inherits from Agent)


    The Bee class has these methods:

        __init__
        update_children_positions
        step_sensors
        integrate
        control
        step_actuators
        pygame_draw - no need to unit test
        __pygame_draw_wings - no need to unit test
        __pygame_draw_wing - no need to unit test
        wing_ends - no need to unit test


        (plus what it inherits from Agent)

'''
