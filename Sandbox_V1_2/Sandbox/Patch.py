from .base import *

from typing import List

class Patch:

    def __init__(self, x: float,
                 y_bottom: float,
                 y_top: float):
        self.x = x
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.length = y_top - y_bottom

    def draw(self, ax) -> None:
        ax.plot([self.x, self.x], [self.y_bottom, self.y_top], linewidth=2, color='white')

class Wall:

    def __init__(self, x: float):
        self.x = x
        self.y_bottom = 1E6
        self.y_top = -1E6
        self.patches: List[Patch] = []

    def add_patch(self, y_top: float, y_bottom: float) -> None:
        self.patches.append(Patch(x=self.x, y_bottom=y_bottom, y_top=y_top))
        if y_bottom < self.y_bottom:
            self.y_bottom = y_bottom
        if y_top > self.y_top:
            self.y_top = y_top

    def draw(self, ax) -> None:
        ax.plot([self.x, self.x], [self.y_bottom, self.y_top], linewidth=2, color='black')
        for patch in self.patches:
            patch.draw(ax)
