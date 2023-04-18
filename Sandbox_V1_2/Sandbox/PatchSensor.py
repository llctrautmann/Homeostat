from .base import *
from .sensors import *

from typing import Tuple

class PatchSensor(Sensor, FOV_thing):
    # construct sensor
    def __init__(self, wall, x: float, y: float, theta: float=0, FOV: float=0.1*math.pi, noisemaker=None, enabled: bool=True):
        super().__init__(x=x, y=y, theta=theta, enabled=enabled)
        self.wall = wall  # a list of Patch instances which this sensor can detect
        self.activation = 0  # sensor activation. this variable is updated in and returned from the step method. it is stored separately in case you want to access it multiple times between simulation steps, although that is unlikely to be necessary
        self.activations = [self.activation]  # for plotting and analysis, a sensor keeps a complete record of its activation over time
        self.noisemaker = noisemaker  # noise source
        self.FOV = FOV  # sensor angular field of view

    def step(self, dt: float) -> float:
        super().step(dt)  # call System step method, to store xy-coordinates and theta

        # distance to wall
        x_dist = self.wall.x - self.x
        # edges of FOV
        theta1 = self.theta + self.FOV/2
        theta2 = self.theta - self.FOV/2

        x1 = np.cos(theta1)
        x2 = np.cos(theta2)

        self.activation = 0

        s = np.sign(x_dist)
        if ((np.sign(x1) == s) or (np.sign(x2) == s)):

            # projection of sensor FOV onto wall
            # y_top = self.y + x_dist * np.sin(theta1)
            # y_bottom = self.y + x_dist * np.sin(theta2)
            l1 = x_dist / math.cos(theta1)
            l2 = x_dist / math.cos(theta2)
            y_bottom = self.y + (l1 * math.sin(theta1))
            y_top = self.y + (l2 * math.sin(theta2))

            # print(y_bottom, y_top)

            if y_bottom > y_top:
                # print(y_bottom, y_top)
                temp = y_top
                y_top = y_bottom
                y_bottom = temp
                # print(y_bottom, y_top, '\n')

            proj_length = y_top - y_bottom

            # print('proj',y_bottom, y_top)

            for patch in self.wall.patches:
                # print('patch', patch.y_bottom, patch.y_top)
                l = 0
                # only look if top of patch is above bottom of FOV
                if patch.y_top > y_bottom:
                    # print("yep")
                    # if FOV top edge is inside patch
                    if y_top < patch.y_top:
                        if y_top > patch.y_bottom:
                            # print("yo")
                            # if FOV is entirely inside patch
                            if y_bottom > patch.y_bottom:
                                # visible length
                                l = proj_length
                                # print('all: ', l)
                            # if only top edge of FOV is inside patch
                            else:
                                # visible length
                                l = y_top - patch.y_bottom
                                # print('top only', l)
                    # if only bottom edge of patch is inside FOV
                    elif y_bottom < patch.y_top and y_bottom > patch.y_bottom:
                        # print('uh?', y_bottom, patch.y_bottom, y_top)
                        # visible length
                        l = patch.y_top - y_bottom
                        # print('bottom only', l)
                    # if patch is entirely inside FOV
                    elif y_top > patch.y_top and y_bottom < patch.y_bottom:
                        l = patch.length

                    # increase activation according to ratio between visible patch length and length of projection of sensor onto wall
                    if l:
                        self.activation += l / proj_length
                # else:
                #     print("nope")

        # add noise, if a noisemaker is implemented
        if self.noisemaker != None:
            self.activation += self.noisemaker.step(dt)

        # record activation
        self.activations.append(self.activation)

        # return activation
        return self.activation

    def draw(self, ax) -> None:
        super().draw(ax)
        self.draw_FOV(ax)

    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        """
            A method for drawing the sensor, as a small circle with lines radiating out from its centre to inidicate the sensor's FOV.

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        super().pygame_draw(screen, scale, shiftx, shifty)

        self.pygame_draw_FOV(screen, scale, shiftx, shifty)
