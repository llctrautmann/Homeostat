import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Robot class has these attributes:

        controller
        sensors
        sensor_angles
        initial_sensor_angles
        left_motor, right_motor

    The Robot class has these methods:

        __init__
        update_children_positions
        reset
        step_actuators
        control
        integrate
        step_sensors
        get_data
        draw - no need to unit test this
        __wheel_ends - no need to unit test this
        pygame_draw - no need to unit test this
        __pygame_draw_wheels - no need to unit test this



'''
