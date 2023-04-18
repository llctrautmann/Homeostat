from .base import *

import math

class HighPassFilter:

    def __init__(self, fc: float):
        self.fc = fc
        self.inputs = [0.]
        self.outputs = [0.]

    def get_alpha(self, dt: float) -> float:
        c = 2 * math.pi * dt * self.fc
        return 1 / (c + 1)

    def step(self, input: float, dt: float) -> float:
        alpha = self.get_alpha(dt)
        o = alpha*(self.outputs[-1] + input - self.inputs[-1])
        self.outputs.append(o)
        self.inputs.append(input)
        return o
