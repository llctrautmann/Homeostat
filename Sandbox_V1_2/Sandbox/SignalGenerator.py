from .base import *
from .stimuli import *
from .noise import *
import copy as cp

import math

class SignalGenerator(Stimulus):

    def __init__(self, freq:float, amp:float, offset: float=0, form: str='square', phase_shift: float=0, is_on: bool=True, noisemaker: NoiseSource=None):
        super().__init__(is_on=is_on)
        self.t = 0
        self.form = form
        self.freq = freq
        self.phase_shift = phase_shift
        self.offset = offset
        self.amp = amp
        self.noisemaker = noisemaker
        self.output = 0
        self.initial_output = self.output
        self.outputs = [self.output]
        self.initial_outputs = [self.output]

    def reset(self):
        super().reset()
        self.output = self.initial_output
        self.outputs = cp.deepcopy(self.initial_outputs)
        if self.noisemaker:
            self.noisemaker.reset()

    def get_data(self) -> dict:
        data = super().get_data()
        data["outputs"] = self.outputs
        return data

    def step(self, dt):
        self.t += dt

        self.output = 0
        if self.form == 'square':
            self.output = self.square()
        elif self.form == 'sine':
            self.output = self.sine()
        elif self.form == 'triangle':
            self.output = self.triangle()
        elif self.form == 'sawtooth':
            self.output = self.sawtooth()

        if self.noisemaker:
            self.output += self.noisemaker.step(dt)

        self.outputs.append(self.output)

        return self.output

    def square(self):
        o = self.plain_sine()
        if o > 0:
            o = 1
        elif o < 0:
            o = -1
        return self.offset + (self.amp * o)

    def sine(self):
        return self.offset + (self.amp * self.plain_sine())

    def plain_sine(self):
        return math.sin(2 * math.pi * self.freq * (self.t + self.phase_shift))

    def triangle(self):
        return self.offset + (self.amp * np.abs((2 * self.plain_sawtooth()) - 1))

    def sawtooth(self):
        return self.offset + (self.amp * self.plain_sawtooth())

    def plain_sawtooth(self):
        t_p = self.freq * (self.t + self.phase_shift)
        return (t_p % 1)
