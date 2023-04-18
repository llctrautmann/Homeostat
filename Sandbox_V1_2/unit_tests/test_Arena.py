import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Arena class has these attributes:

        agents
        x_left
        x_right
        y_top
        y_bottom

    The Arena class has these methods:

        __init__
        step
        move
        draw - no need to unit test
        draw2 - no need to unit test
        pygame_draw - no need to unit test
        line_drawer - no need to unit test

'''

class Test_Arena(MyTestCase):

    def test_init(self) -> None:

        print("Testing Arena")

        a = Arena(agents=[], x_left=-1, x_right=8, y_top=10, y_bottom=0)
        self.assertTrue(a.agents == [])
        self.assertTrue(a.x_left == -1)
        self.assertTrue(a.x_right == 8)
        self.assertTrue(a.y_top == 10)
        self.assertTrue(a.y_bottom == 0)

    def test_step(self) -> None:

        pass

    def test_move(self) -> None:

        pass
