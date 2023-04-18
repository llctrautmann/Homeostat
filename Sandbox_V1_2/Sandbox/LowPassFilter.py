from .base import *

import math

class LowPassFilter:

    def __init__(self, fc: float):
        self.fc = fc
        self.outputs = [0.]

    def get_alpha(self, dt: float) -> float:
        c = 2 * math.pi * dt * self.fc
        return c / (c + 1)

    def step(self, input: float, dt: float) -> float:
        alpha = self.get_alpha(dt)
        o = self.outputs[-1] - alpha*(self.outputs[-1] - input)
        self.outputs.append(o)
        return o
