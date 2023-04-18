import sys
# relative path to folder which contains SituSim_V2
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The NoiseSource class has these attributes:

        noise
        noises

    The NoiseSource class has these methods:

        __init__
        step
        reset
        get_data

'''

'''

    The WhiteNoiseSource class has these attributes:

        extent
        min_val

        (plus what it inherits from NoiseSource)

    The WhiteNoiseSource class has these methods:

        __init__
        step

        (plus what it inherits from NoiseSource)

'''

'''

    The BrownNoiseSource class has these attributes:

        max_step_size

        (plus what it inherits from NoiseSource)

    The BrownNoiseSource class has these methods:

        __init__
        step

        (plus what it inherits from NoiseSource)

'''

'''

    The SpikeNoiseSource class has these attributes:

        prob
        pos_size
        neg_size

        (plus what it inherits from NoiseSource)

    The SpikeNoiseSource class has these methods:

        __init__
        step
        set_params

        (plus what it inherits from NoiseSource)

'''

'''

    The NoiseMaker class has these attributes:

        noise_sources

        (plus what it inherits from NoiseSource)

    The NoiseMaker class has these methods:

        __init__
        step

        (plus what it inherits from NoiseSource)

'''

class Test_Noise(MyTestCase):

    def test_init(self) -> None:
        print("Testing Noise")
        n = WhiteNoiseSource(min_val=-1, max_val=1)
        l1 = [0, 1, 2, 3, 4, 5]
        n.noises = l1

        self.assertTrue(n.noises == l1)

        d = n.get_data()
        n.reset()

        self.assertFalse(n.noises == l1)
        self.assertTrue(d["noises"] == l1)
