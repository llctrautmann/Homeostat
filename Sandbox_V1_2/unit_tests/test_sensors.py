import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Sensor class has these attributes:

        colour
        radius
        enabled

        (plus what it inherits from System)

    The Sensor class has these methods:

        __init__
        get_data
        draw - no need to unit test
        pygame_draw - no need to unit test

        (plus what it inherits from System)

'''


'''

    The LightSensor class has these attributes:

        light_sources
        initial_light_sources
        activation
        activations
        noisemaker
        FOV
        label
        initial_label
        initial_enabled
        initial_FOV

        (plus what it inherits from Sensor)

    The LightSensor class has these methods:

        __init__
        reset
        step
        get_data
        pygame_draw

        (plus what it inherits from Sensor)

'''

'''

    FOV_thing only exsits for drawing sensor fields of view.
    This functionality has its own class so that it can be inherited by
    any object that needs it - I'm reluctant to make it part of Sensor, as
    not all subclasses of Sensor will necessarily have a FOV.

    There isn't much to unit test here, as it is only for visualisation -
    it is highly unlikely that errors with this code wouldn't be easy to
    see during animation.

'''

class Test_Sensor(MyTestCase):

    def test_init(self) -> None:

        # TEST SENSOR CLASS

        # TEST STEP METHOD

        # TEST RESET METHOD



        # TEST LIGHTSENSOR CLASS
        s = LightSensor(x=0, y=2, light_sources=[])

        ls = LightSource(x=0, y=0)
        s = LightSensor(x=0, y=2, light_sources=[ls])

        # TEST STEP METHOD

        # TEST RESET METHOD
