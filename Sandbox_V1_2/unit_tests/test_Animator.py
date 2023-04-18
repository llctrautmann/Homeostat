import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Animator class has these attributes:

        screen_width
        screen
        paused
        delay
        systems
        animate_current
        animate_any

    The Animator class has these methods:

        __init__
        draw_frame - can't really unit test, and don't need to - test by running
        stop - can't really unit test, and don't need to - test by running
        shutdown - can't really unit test, and don't need to - test by running
        write_title - can't really unit test, and don't need to - test by running

'''

class Test_Animator(MyTestCase):

    def test_init(self) -> None:

        print("Testing Animator")

        a = Animator(systems=[])
        self.assertTrue(a.systems == [])
        self.assertTrue(a.screen_width == 700)
        self.assertFalse(a.paused)
        self.assertTrue(a.delay == 0)

        a = Animator(systems=[], screen_width=1200, paused=True, delay=100)
        self.assertTrue(a.systems == [])
        self.assertTrue(a.screen_width == 1200)
        self.assertTrue(a.paused)
        self.assertTrue(a.delay == 100)
