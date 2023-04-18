import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Simulator class has these attributes:

        agents
        envs
        disturbances
        obj_fun
        t, ts
        run_completed
        duration
        dt

    The Simulator class has these methods:

        get_systems
        reset
        get_data
        step_forwards


'''
