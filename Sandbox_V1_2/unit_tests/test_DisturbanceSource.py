import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The DisturbanceSource class has these attrbutes:

        enabled
        t
        start_times
        stop_times
        init_enabled
        init_start_times
        init_stop_times

    The DisturbanceSource class has these methods:

        __init__
        step
        reset


    The MovingSensorsDisturbanceSource class has these attrbutes:

        robot
        move_left
        move_right
        noisesource
        sensor_indices

        (plus what it inherits from DisturbanceSource)

    The MovingSensorsDisturbanceSource class has these methods:

        __init__
        step
        reset

    The SensoryInversionDisturbanceSource class has these attrbutes:

        robot
        sensor_indices

        (plus what it inherits from DisturbanceSource)

    The SensoryInversionDisturbanceSource class has these methods:

        __init__
        step


'''
