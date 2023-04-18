from .base import *
from .stimuli import *
from .noise import *

import math

# base sensor class. in the current implementation, only contains methods for drawing
class Sensor(System):
    """
        An abstract class for representing sensors.
    """
    # by default, a Sensor has no position, but one can be specified (and for most sensors will)
    def __init__(self, x: float=None, y: float=None, theta: float=None, colour: str='red', radius: float =0.2, enabled: bool=True):
        '''
            __init__(x: float=None, y: float=None, theta: float=None, colour: str='red', radius: float =0.2, enabled: bool=True)

            :param x: The initial x-coordinate of the :class:`Sensor`, defaults to ``None``.
            :type x: float

            :param y: The initial y-coordinate of the :class:`Sensor`, defaults to ``None``.
            :type y: float

            :param theta:The initial angular orientation of the :class:`Sensor`, defaults to ``None``.
            :type theta: float

            :param colour: The colour of the sensor, for drawing.
            :type colour: str

            :param radius: The radius of the sensor, for drawing.
            :type radius: float

            :param enabled: A flag for specifying whether or not a sensor is enabled. Only the attribute is implemented here - how to use it is a decision for subclasses, e.g. in :class:`LightSensor`.
            :type enabled: bool
        '''
        super().__init__(x, y, theta)
        self.colour = colour
        self.radius = radius
        self.enabled = enabled

    # draw sensor in the specified matplotlib axes
    def draw(self, ax) -> None:
        """
            A method to draw the sensor in the specified matplotlib axes, as a small coloured circle.

            :param ax:
            :type ax: Matplotlib axes
        """
        if self.has_position:
            ax.add_artist(mpatches.Circle((self.x, self.y), self.radius, color=self.colour))
            ax.plot(self.x, self.y, 'k.')

    # draw sensor in a pygame display
    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        """
            A method to draw the sensor in the specified PyGame display, as a small coloured circle.

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        if self.has_position:
            pygame.draw.circle(screen, center=(scale*self.x+shiftx, scale*self.y+shifty), color=self.colour, radius=scale*self.radius)

    def get_data(self) -> Dict[str, Union[float, List[float], str]]:
        """
            A method to get the sensors data, in the form of a dict.

            :return: The sensor's data, which includes the data returned from :meth:`Sandbox.System.get_data`, as well as the sensor's colour and radius (which are both assumed to be static).
            :rtype: dict
        """
        data = super().get_data()
        data["colour"] = self.colour
        data["radius"] = self.radius
        return data

class FOV_thing:
    """
        A class which is used only for drawing a sensor's field of view (FOV).
        This code is separated from :class:`Sensor`, as not all sensors will necesarily have a FOV. Sensor classes which have FOV should have multiple inheritance of :class:`Sensor` and :class:`FOV_thing`, as in the case of :class:`LightSensor`.

        NOTE: this class needs code adding for Matplotlib drawing - it currently only has methods for PyGame.
    """
    def pygame_draw_FOV(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        """
            A method to draw a FOV in the specified PyGame display, with two short lines which indicate its angular extent.
            NOTE: this method is in need of some improvement, in the next implementation - it should be possible to draw the lines at different lengths, and it should also somehow be made clear which line starts and which ends the FOV.

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        left_end_x, left_end_y, right_end_x, right_end_y = self.__fov_ends()

        pygame.draw.line(screen, color='green',
                         start_pos=(scale * self.x + shiftx, scale * self.y + shifty),
                         end_pos=(scale * left_end_x + shiftx, scale * left_end_y + shifty), width=2)
        pygame.draw.line(screen, color='green',
                         start_pos=(scale * self.x + shiftx, scale * self.y + shifty),
                         end_pos=(scale * right_end_x + shiftx, scale * right_end_y + shifty), width=2)

    # calculate end coords of lines indicating field of view
    def __fov_ends(self) -> Tuple[float, float, float, float]:
        '''
            A method for calculating the coordinates of the ends of the lines for drawing a FOV.
        '''
        left_end_x = self.x + math.cos(self.theta + self.FOV/2)
        left_end_y = self.y + math.sin(self.theta + self.FOV/2)
        right_end_x = self.x + math.cos(self.theta - self.FOV/2)
        right_end_y = self.y + math.sin(self.theta - self.FOV/2)
        return left_end_x, left_end_y, right_end_x, right_end_y

    # draw lines indicating field of view in matplotlib axes
    def draw_FOV(self, ax) -> None:
        left_end_x, left_end_y, right_end_x, right_end_y = self.__fov_ends()
        ax.plot([self.x, left_end_x],
                 [self.y, left_end_y], 'b--', linewidth='2')
        ax.plot([self.x, right_end_x],
                 [self.y, right_end_y], 'r--', linewidth='2')

# a class to define a sensor which detects instances of the LightSource class
class LightSensor(Sensor, FOV_thing):
    """
        A class which represents a light sensor. :class:`LightSensor` inherits both from :class:`Sensor` and :class:`FOV_thing`.
    """
    def __init__(self, light_sources: List[LightSource], x: float, y: float, theta: float=0, FOV: float=2*math.pi, noisemaker: NoiseSource=None, label: str=None, enabled: bool=True):
        '''
            __init__(light_sources: List[LightSource], x: float, y: float, theta: float=0, FOV: float=2*math.pi, noisemaker: NoiseSource=None, label: str=None, enabled: bool=True)

            :param light_sources:
            :type light_sources: list[:class:`Light_Source`]

            :param x:
            :type x: float

            :param y:
            :type y: float

            :param theta:
            :type theta: float

            :param FOV:
            :type FOV: float

            :param noisemaker:
            :type noisemaker: :class:`NoiseSource`

            :param label: A :class:`LightSensor`'s label determines which of the :class:`LightSource` s in its list it can actually detect. Defaults to ``None``, in which case the sensor will detect all light sources in its list. If a sensor's ``label`` attribute is set to some string, then it will only detect light sources which have their ``label`` attributes set to the same value.
            :type label: str

            :param enabled: A flag for specifying whether or not a light sensor is enabled. Defaults to ``True``. If set to ``False``, then the sensor will not detect anything.
            :type enabled: bool
        '''
        super().__init__(x=x, y=y, theta=theta, enabled=enabled)
        self.light_sources = light_sources
        self.initial_light_sources = light_sources  # a list of LightSource instances which this sensor can detect
        self.activation: float = 0.0  # sensor activation. this variable is updated in and returned from the step method. it is stored separately in case you want to access it multiple times between simulation steps, although that is unlikely to be necessary
        self.activations = [self.activation]  # for plotting and analysis, a sensor keeps a complete record of its activation over time
        self.noisemaker = noisemaker  # noise source
        self.FOV = FOV  # sensor angular field of view

        self.label = label
        self.initial_label = label
        self.initial_enabled = enabled # Perhaps should be moved to Sensor!?
        self.initial_FOV = FOV

    def reset(self) -> None:
        '''
            A method to reset a sensor to its initial state, by resetting its ``x``, ``y``, ``theta``, ``light_sources``, ``activation``, ``activations``, ``label``, ``enabled``, and ``FOV`` attributes to their values at time of construction. If the sensor has a ``noisemaker``, then the reset method of that object will also be called.

            Note: this method will not reset any attributes which have been added outside of the :meth:`Sandbox.LightSensor.__init__` method.
        '''
        super().reset()
        self.activation = 0
        self.activations = [self.activation]
        self.label = self.initial_label
        self.enabled = self.initial_enabled
        self.FOV = self.initial_FOV
        self.light_sources = self.initial_light_sources
        if self.noisemaker:
            self.noisemaker.reset()

    def step(self, dt: float) -> float:
        """
            A method to step a light sensor forwards in time. A :class:`LightSensor` has no dynamics, so technically is not stepped in time, but 'step' is used for consistency.

            :param dt: Integration interval - not used here.
            :type dt: float

            :return: The activation (level of stimulation) of the sensor. When the sensor detects multiple light sources, their effects are summed linearly.
            :rtype: float
        """
        super().step(dt)  # call System step method, to store xy-coordinates and theta
        self.activation = 0.0  # begin with zero activation, and add to it for every detected light source
        # only detect anything if enabled
        if self.enabled:
            for source in self.light_sources:  # for every light source the sensor can detect
                # if this sensor has a label set, then it will only detect sensors with the same label
                if not self.label or self.label == source.label:
                    angle_to_source = math.atan2(source.y - self.y, source.x - self.x)  # find angle of vector from light source to sensor
                    if abs(angle_difference(angle_to_source, self.theta)) <= (self.FOV/2):  # if angle is within field fo view, the sensor detects the light
                        self.activation += source.get_brightness_at(self.x,self.y)  # stimuli from multiple lights are added linearly

            # add noise, if a noisemaker is implemented
            if self.noisemaker != None:
                self.activation += self.noisemaker.step(dt)

        # record activation
        self.activations.append(self.activation)  # store activation

        # return activation
        return self.activation  # return activation

    ##
    #
    # This method allows the same sensor to be used in multiple runs, and collect
    # its data each time
    #
    def get_data(self) -> dict:
        """

        """
        data = super().get_data()
        data["activations"] = self.activations
        data["FOV"] = self.FOV
        data["label"] = self.label
        data["noises"] =  None
        if self.noisemaker:
            # print(self.noisemaker.get_data())
            # print()
            data["noises"] = self.noisemaker.get_data()["noises"]
        return data

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
