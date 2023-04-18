import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The System class has these attributes:

        x, y, theta
        xs, ys, thetas
        has_position, has_orientation

        iff a System has_position, then it also has x, y, xs, ys
        iff a System has_orientation, then is also has theta, thetas

        if a value for either of x or y is passed in to __init__, then the System has position
        if a value for theta is passed in to __init__, the the System has orientation

        There are some systems which do not have position or orientation, e.g. Controllers and NoiseSources

    The System class has these methods:

        __init__
        step
        get_data
        reset
        get_data_and_reset
        init_xy
        init_theta

'''

class Test_System(MyTestCase):

    def test_init(self) -> None:

        print("Testing Sytem")

        s = System(x=0, y=0, theta=0)
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertTrue(hasattr(s, 'thetas'))
        self.assertTrue(s.has_position)
        self.assertTrue(s.has_orientation)

        s = System(x=0, y=0)
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertFalse(hasattr(s, 'thetas'))
        self.assertTrue(s.has_position)
        self.assertFalse(s.has_orientation)

        s = System(x=0)
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertFalse(hasattr(s, 'thetas'))
        self.assertTrue(s.has_position)
        self.assertFalse(s.has_orientation)

        s = System(y=0)
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertFalse(hasattr(s, 'thetas'))
        self.assertTrue(s.has_position)
        self.assertFalse(s.has_orientation)

        s = System(theta=0)
        self.assertFalse(hasattr(s, 'xs'))
        self.assertFalse(hasattr(s, 'ys'))
        self.assertTrue(hasattr(s, 'thetas'))
        self.assertFalse(s.has_position)
        self.assertTrue(s.has_orientation)

    def test_step(self):

        s = System(x=0, y=0, theta=0)
        n = 10
        for _ in range(n):
            s.step(1)
        self.assertTrue(len(s.xs) == n+1)
        self.assertTrue(len(s.ys) == n+1)
        self.assertTrue(len(s.thetas) == n+1)

    def test_get_data_and_reset(self) -> None:

        s = System(x=0, y=0, theta=0)
        l1 = [0, 1, 2, 3, 4, 5]
        l2 = [0, 6, 7, 8, 9, 10]
        l3 = [0, 5, 4, 3, 2, 1]
        for i in range(5):
            s.xs.append(l1[i+1])
            s.ys.append(l2[i+1])
            s.thetas.append(l3[i+1])

        self.assertTrue(s.xs == l1)
        self.assertTrue(s.ys == l2)
        self.assertTrue(s.thetas == l3)

        d = s.get_data()
        s.reset()

        self.assertFalse(s.xs == l1)
        self.assertFalse(s.ys == l2)
        self.assertFalse(s.thetas == l3)

        # check that the reset method has not changed the data returned from get_data()
        self.assertTrue(d['xs'] == l1)
        self.assertTrue(d['ys'] == l2)
        self.assertTrue(d['thetas'] == l3)

        s = System(x=0, y=0, theta=0)
        l1 = [0, 1, 2, 3, 4, 5]
        l2 = [0, 6, 7, 8, 9, 10]
        l3 = [0, 5, 4, 3, 2, 1]
        for i in range(5):
            s.xs.append(l1[i+1])
            s.ys.append(l2[i+1])
            s.thetas.append(l3[i+1])

        d = s.get_data_and_reset()

        self.assertFalse(s.xs == l1)
        self.assertFalse(s.ys == l2)
        self.assertFalse(s.thetas == l3)

        # check that the reset method has not changed the data returned from get_data()
        self.assertTrue(d['xs'] == l1)
        self.assertTrue(d['ys'] == l2)
        self.assertTrue(d['thetas'] == l3)
