import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Agent class has these attributes:

        colour
        radius
        light

        (plus what it inherits from System)

    The Agent class has these methods:

        __init__
        step - can't be tested in Agent - requires methods which are implemented in subclasses
        push - can't be tested in Agent - requires methods which are implemented in subclasses

'''

class Test_Agent(MyTestCase):

    def test_init(self) -> None:

        print("Testing Agent")

        x_co = 3
        y_co = 2
        a = Agent(x=x_co, y=y_co, colour='green')
        self.assertTrue(a.x == x_co)
        self.assertTrue(a.y == y_co)
        self.assertTrue(a.colour == 'green')
        self.assertTrue(a.xs == [x_co])
        self.assertTrue(a.ys == [y_co])
        self.assertFalse(hasattr(a, 'theta'))
        self.assertFalse(hasattr(a, 'thetas'))
        self.assertTrue(a.has_position)
        self.assertFalse(a.has_orientation)
        self.assertTrue(a.radius == 1)
        self.assertTrue(a.light == None)

        a = Agent(x=x_co, y=y_co, colour='green', theta=6, radius=1.2)
        self.assertTrue(a.x == x_co)
        self.assertTrue(a.y == y_co)
        self.assertTrue(a.colour == 'green')
        self.assertTrue(a.xs == [x_co])
        self.assertTrue(a.ys == [y_co])
        self.assertTrue(a.theta == 6)
        self.assertTrue(a.thetas == [6])
        self.assertTrue(a.has_position)
        self.assertTrue(a.has_orientation)
        self.assertTrue(a.radius == 1.2)
