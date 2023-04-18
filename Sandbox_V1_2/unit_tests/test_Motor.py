import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Motor class has these attributes:

        noisemaker
        speed
        speeds
        motor_inertia_coeff
        max_speed
        reversed
        reverseds

    The Motor class has these methods:

        __init__
        step
        reset
        get_data

'''

class Test_Motor(MyTestCase):

    def test_init(self) -> None:

        print("Testing Motor")

        m = Motor(max_speed=2)
        self.assertTrue(not hasattr(m, 'x'))
        self.assertTrue(not hasattr(m, 'y'))
        self.assertTrue(not hasattr(m, 'theta'))
        self.assertTrue(not hasattr(m, 'xs'))
        self.assertTrue(not hasattr(m, 'ys'))
        self.assertTrue(not hasattr(m, 'thetas'))
        self.assertTrue(m.max_speed==2)
        self.assertTrue(m.motor_inertia_coeff==1)
        self.assertTrue(not m.reversed)
        self.assertTrue(not m.noisemaker)

        m = Motor(max_speed=-2)
        self.assertTrue(m.max_speed==2)

        m = Motor(max_speed=2, motor_inertia_coeff=1)
        self.assertTrue(m.motor_inertia_coeff==2)

        m = Motor(max_speed=2, motor_inertia_coeff=-1)
        self.assertTrue(m.motor_inertia_coeff==1)

    def test_step(self) -> None:

        m = Motor(max_speed=2, reversed=True)
        self.assertTrue(m.reversed)
        n = 10
        for _ in range(n):
            m.step(2, 0.1)

        self.assertTrue(len(m.speeds) == n+1)

    def test_get_data_and_reset(self) -> None:

        m = Motor(max_speed=2, reversed=True)
        l1 = [0, 1, 2, 3, 4, 5]
        m.speeds = l1

        self.assertTrue(m.speeds == l1)

        d = m.get_data()
        m.reset()

        self.assertFalse(m.speeds == l1)

        # check that the reset method has not changed the data returned from get_data()
        self.assertTrue(d['speeds'] == l1)

        coeff = 1
        max_s = 5
        m = Motor(max_speed=max_s, motor_inertia_coeff=coeff)
        self.assertTrue(m.motor_inertia_coeff==coeff+1)
        self.assertTrue(m.max_speed==max_s)

        l1 = [0, 1, 2, 3, 4, 5]
        m.speeds = l1

        m.motor_inertia_coeff = 10
        m.max_speed = 9

        d = m.get_data_and_reset()
        self.assertFalse(m.speeds == l1)

        # check that the reset method has not changed the data returned from get_data()
        self.assertTrue(d['speeds'] == l1)
        self.assertTrue(m.motor_inertia_coeff==coeff+1)
        self.assertTrue(m.max_speed==max_s)

    def step(self):

        m = Motor(max_speed=-2)
        for _ in range(n):
            m.step(1, 10)
        self.assertTrue(len(s.speeds) == n+1)
        self.assertTrue(len(s.reverseds) == n+1)
