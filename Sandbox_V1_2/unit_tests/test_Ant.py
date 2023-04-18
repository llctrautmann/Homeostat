import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The AntController class has these attributes:

        theta_noisemaker
        speed_noisemaker

        inputs

        theta_command
        speed_command
        theta_commands
        speed_commands

        step_fun

        (plus what it inherits from Controller)

    The AntControllerator class has these methods:

        __init__
        step

        (plus what it inherits from Controller)

'''

'''

    The Ant class has these attributes:

        sensors
        sensor_angles
        initial_sensor_angles

        max_speed
        theta_inertia
        speed_inertia

        controller

        speed
        speeds

        (plus what it inherits from Agent)


    The Ant class has these methods:

        __init__
        update_children_positions
        step_sensors
        integrate
        control
        step_actuators
        pygame_draw - no need to unit test
        __pygame_draw_legs - no need to unit test
        leg_ends - no need to unit test


        (plus what it inherits from Agent)

'''
