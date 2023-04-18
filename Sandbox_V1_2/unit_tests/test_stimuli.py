import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Stimulus class has these attributes:

        is_on

    The Stimulus class has these methods:

        __init__
        get_distance
        get_data
        reset

    The LightSource class has these attributes:

        brightness
        gradient
        model
        colour
        label

    The LightSource class has these methods:

        __init__
        draw - no need to unit test
        pygame_draw - no need to unit test
        get_brightness_at
        linear_model
        inv_sq_model
        get_data
        reset



'''

class Test_Stimulus(MyTestCase):

    def test_init(self) -> None:

        print("Test Stimulus")

        s = Stimulus()
        self.assertTrue(not hasattr(s, 'x'))
        self.assertTrue(not hasattr(s, 'y'))
        self.assertTrue(not hasattr(s, 'theta'))
        self.assertTrue(not hasattr(s, 'xs'))
        self.assertTrue(not hasattr(s, 'ys'))
        self.assertTrue(not hasattr(s, 'thetas'))
        self.assertTrue(s.is_on)

        s = Stimulus(x=1, y=2, is_on=False)
        self.assertTrue(hasattr(s, 'x'))
        self.assertTrue(hasattr(s, 'y'))
        self.assertTrue(not hasattr(s, 'theta'))
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertTrue(not hasattr(s, 'thetas'))
        self.assertTrue(not s.is_on)

        s = Stimulus(x=1, y=2, theta=1)
        self.assertTrue(hasattr(s, 'x'))
        self.assertTrue(hasattr(s, 'y'))
        self.assertTrue(hasattr(s, 'theta'))
        self.assertTrue(hasattr(s, 'xs'))
        self.assertTrue(hasattr(s, 'ys'))
        self.assertTrue(hasattr(s, 'thetas'))

        s = Stimulus(theta=1)
        self.assertTrue(not hasattr(s, 'x'))
        self.assertTrue(not hasattr(s, 'y'))
        self.assertTrue(hasattr(s, 'theta'))
        self.assertTrue(not hasattr(s, 'xs'))
        self.assertTrue(not hasattr(s, 'ys'))
        self.assertTrue(hasattr(s, 'thetas'))

        # causes unittest failure, as hinted types are not enforced
        # self.assertRaises(Exception, Stimulus, x='uh', y='oh')

        # correctly picked up by mypy as having wrong types
        # s = Stimulus(x='uh', y='oh')


    def test_get_distance(self) -> None:

        s = Stimulus(theta=1)
        self.assertTrue(s.get_distance(0,0) is None)

        s = Stimulus(x=0, y=0, theta=1)
        self.assertNear(s.get_distance(1,1), np.sqrt(2))

        s = Stimulus(x=10, y=0, theta=1)
        self.assertNotNear(s.get_distance(1,1), np.sqrt(2))

    def test_get_data_and_reset(self):

        s = Stimulus(x=0, y=10, theta=1)
        s.is_on = False
        s.x = 2
        s.y = 1
        s.theta = -4
        d = s.get_data_and_reset()
        self.assertTrue(d["x"] == 2)
        self.assertTrue(d["y"] == 1)
        self.assertTrue(d["theta"] == -4)
        self.assertTrue(d["is_on"] == False)

        self.assertTrue(s.x == 0)
        self.assertTrue(s.y == 10)
        self.assertTrue(s.theta == 1)
        self.assertTrue(s.is_on == True)

class Test_LightSource(MyTestCase):

    def test_init(self) -> None:

        print("Test LightSource")

        # causes unittest failure, as LightSource(x=0, y=0) is fine
        # self.assertRaises(Exception, LightSource, x=0, y=0)

        # LightSource with no coords causes exception in __init__
        self.assertRaises(Exception, LightSource)

        ls = LightSource(x=0, y=0)
        self.assertTrue(hasattr(ls, 'x'))
        self.assertTrue(hasattr(ls, 'y'))
        self.assertTrue(not hasattr(ls, 'theta'))
        self.assertTrue(hasattr(ls, 'xs'))
        self.assertTrue(hasattr(ls, 'ys'))
        self.assertTrue(not hasattr(ls, 'thetas'))
        self.assertTrue(ls.is_on)
        self.assertTrue(ls.brightness == 1)
        self.assertTrue(ls.gradient == 0.01)
        self.assertTrue(ls.model == 'inv_sq')
        self.assertTrue(ls.colour == 'yellow')
        self.assertTrue(ls.label == None)

    def test_models(self):

        ls = LightSource(x=0, y=0, model='inv_sq')
        bs = []
        for r in range(10):
            bs.append(ls.get_brightness_at(0, r))
            if r > 0:
                self.assertTrue(bs[-1] < bs[-2])

        ls = LightSource(x=0, y=0, model='linear')
        bs = []
        for r in range(10):
            bs.append(ls.get_brightness_at(0, r))
            if r > 0:
                self.assertTrue(bs[-1] < bs[-2])

        ls = LightSource(x=0, y=0, model='binary')
        bs = []
        for r in range(10):
            bs.append(ls.get_brightness_at(0, r))
            if r > 0:
                self.assertTrue(bs[-1] == bs[-2])

    def test_get_data_and_reset(self):

        ls = LightSource(x=0, y=2, model='inv_sq')
        self.assertTrue(ls.x == 0)
        self.assertTrue(ls.y == 2)
        self.assertTrue(not hasattr(ls, 'theta'))
        self.assertTrue(hasattr(ls, 'xs'))
        self.assertTrue(hasattr(ls, 'ys'))
        self.assertTrue(not hasattr(ls, 'thetas'))
        self.assertTrue(ls.is_on)
        self.assertTrue(ls.brightness == 1)
        self.assertTrue(ls.gradient == 0.01)
        self.assertTrue(ls.model == 'inv_sq')
        self.assertTrue(ls.colour == 'yellow')
        self.assertTrue(ls.label == None)

        ls.x = 78
        ls.y = 97
        ls.is_on = False
        ls.brightness = 29
        ls.gradient = 0.5
        ls.model = 'binary'
        ls.colour = 'red'
        ls.label = 'red'

        d = ls.get_data_and_reset()

        self.assertTrue(d["x"] == 78)
        self.assertTrue(d["y"] == 97)
        self.assertTrue(d["is_on"] == False)
        self.assertTrue(d["brightness"] == 29)
        self.assertTrue(d["gradient"] == 0.5)
        self.assertTrue(d["model"] == 'binary')
        self.assertTrue(d["colour"] == 'red')
        self.assertTrue(d["label"] == 'red')

        self.assertTrue(ls.x == 0)
        self.assertTrue(ls.y == 2)
        self.assertTrue(not hasattr(ls, 'theta'))
        self.assertTrue(hasattr(ls, 'xs'))
        self.assertTrue(hasattr(ls, 'ys'))
        self.assertTrue(not hasattr(ls, 'thetas'))
        self.assertTrue(ls.is_on)
        self.assertTrue(ls.brightness == 1)
        self.assertTrue(ls.gradient == 0.01)
        self.assertTrue(ls.model == 'inv_sq')
        self.assertTrue(ls.colour == 'yellow')
        self.assertTrue(ls.label == None)
