from .base import *

from .LowPassFilter import *
from .HighPassFilter import *

from typing import List

class HRCorrelator:

    def __init__(self, hp_fc: float, lp_fc: float):
        self.hpf1 = HighPassFilter(hp_fc)
        self.hpf2 = HighPassFilter(hp_fc)
        self.lpf = LowPassFilter(lp_fc)
        self.outputs: List[float] = [0.]

    def step(self, input1: float, input2: float, dt: float) -> float:
        i1 = self.lpf.step(self.hpf1.step(input1, dt), dt)
        i2 = self.hpf2.step(input2, dt)
        output = i1 * i2
        self.outputs.append(output)

        return output
