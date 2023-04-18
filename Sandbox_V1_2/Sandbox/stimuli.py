from .System import *

'''
    Note: Stimulus classes are not normally stepped, so
    they don't keep histories.
'''

# this is the base class for stimuli in the environment. it could be used in other ways, but for now is only used as the
# base for the various types of light sources
class Stimulus(System):
    """
        An abstract class to represent a source of sensory stimulation.
    """
    def __init__(self, x: float=None, y: float=None, theta: float=None, is_on: bool=True):
        """
            __init__(x: float=None, y: float=None, theta: float=None, is_on: bool=True)

            :param x: The x-coordinate of the stimulus. Defaults to ``None``.
            :type x: float

            :param y: The y-coordinate of the stimulus. Defaults to ``None``.
            :type y: float

            :param theta: The angular orientation of the stimulus. Defaults to ``None``.
            :type theta: float

            :param is_on: A flag which can be use to determine whether or not a stimulus can be detected. Defaults to ``True``. This allows for the stimulus to be turned on and off, but the response to the state of this flag is determined elsewhere, e.g. in :class:`LightSource` - a sensor implementation can choose to ignore this flag and detect a stimulus even when it is not on.
            :type is_on: bool
        """
        super().__init__(x, y, theta)
        self.is_on = is_on
        self.initial_is_on = is_on

    # get distance from the given xy coordinates to the Stimulus, if the Stimulus has position
    def get_distance(self, x: float, y: float) -> float:
        """
            A method for finding the Euclidean distance from the given xy-coordinates to the position of this sensor, assuming it has position - this method should not be invoked on an instance of :class:`Stimulus` which does not have position.

            :param x: The x-component of the position to find the distance from.
            :type x: float

            :param y: The y-component of the position to find the distance from.
            :type y: float

            :return: The distance.
            :rtype: float
        """
        if self.has_position:
            vec = np.array([self.x - x, self.y - y])  # vector from sensor to stimulus
            return np.linalg.norm(vec)  # length of vector, i.e. distance from sensor to stimulus
        else:
            return None

    def get_data(self) -> dict:
        """
            A method to return the data of the stimulus.
            Note: this is not likely to be particularly useful, as it only contains the original state, as recorded by :class:`System`, and the current state of "is_on".

            :return: A dict containing the data returned from :meth:`Sandbox.System.get_data`, plus ``is_on``.
            :rtype: dict
        """
        data = super().get_data()
        data["is_on"] = self.is_on
        return data

    def reset(self) -> None:
        """
            A method to reset the :class:`Stimulus` to its original state.
        """
        super().reset()
        self.is_on = self.initial_is_on

# this class implements a static and noiseless light source. it has two models for decay of brightness over distance,
# inverse square and linear. you should use whichever you find easiest, but that will depend on what kind of controllers
# you are working on
#   in the linear case, brightness decays from the maximum to zero, according to the specified
#   gradient.
#       - the downside to this is that there will be a maximum detection distance, beyond which a sensor will not detect
#       the light, but this is only likely to be a problem if a large decay gradient is used (not that problems are
#       necessarily bad - they can make things more interesting).
#       - the upside to the linear decay model is that it may make it easier to program certain kinds of controller.
#   in the inverse square case, brightness decays in a way which is closer to reality.
#       - the upside to the inverse square model is that there is no hard limit to detection range (although there will
#       be a distance at which a sensor is barely stimulated by it)
#       - the downside is that the inverse square model can make it more difficult to program certain kinds of
#       controller, due to its nonlinearity.
class LightSource(Stimulus):
    """
        A subclass of :class:`Stimulus`, :class:`LightSource` is a class which represents a light source. It is possible to set the "model" of intensity of a :class:`LightSource` to be either an inverse square law of decay with distance, or a linear one, or for the light to be detected with constant brightness regardless of distance.

        .. image:: images/LinearLightSource.svg
          :width: 600
          :align: center
          :alt: Linear light decay model

        .. image:: images/InvSqLightSource.svg
          :width: 600
          :align: center
          :alt: Inverse square light decay model

    """
    # construct light source
    def __init__(self, x: float, y: float, theta: float=None, brightness: float=1, gradient: float=0.01, model: str='inv_sq', is_on: bool=True, colour: str='yellow', label: str=None):
        """
            __init__(x: float=None, y: float=None, theta: float=None, is_on: bool=True)

            :param x: The x-coordinate of the light.
            :type x: float

            :param y: The y-coordinate of the light.
            :type y: float

            :param theta: The orientation of the light. In this implementation, this is unused, as a :class:`LightSource` can be detected from any angle (although the same will not necessarily be true of its subclasses).
            :type theta: float

            :param brightness: The brightness of the light, at its own coordinate.
            :type brightness: float

            :param gradient: The gradient of brightness decay with distance, when the linear model is used.
            :type gradient: float

            :param model: The model. ``inv_sq``, ``linear``, and ``binary`` are valid models.
            :type model: str

            :param colour: The colour of the light. Note: this is for drawing only - no system currently implemented in *Sandbox* can detect a light's colour.
            :type colour: str

            :param label: A light source's label defines a group. Any :class:`LightSensor` with the same label attribute will detect all, and only, light sources in that group. A :class:`LightSensor` with no label will detect any :class:`LightSource` in its list, regardless of label.
            :type label: str
        """
        super().__init__(x, y, theta, is_on)  # call Stimulus constructor
        self.brightness = brightness  # this is the brightness of the light at the source
        self.initial_brightness = brightness

        self.gradient = gradient  # this determines how quickly the brightness decays when the linear model is used
        self.initial_gradient = gradient

        self.model = model  # model can be inv_sq, which is realistic, or linear, which is not physically realistic but is easier to work with
        self.initial_model = model

        self.colour = colour
        self.initial_colour = colour

        self.label = label
        self.initial_label = label

    def draw(self, ax) -> None:
        """
            A method to draw the light source in the specified matplotlib axes.
            A :class:`LightSource` which is switched on is drawn as a circle with colour specified by the light source's ``colour`` attribute, with a smaller circle in its center which is orange coloured.  A :class:`LightSource` which is not on will have its outer circle coloured in grey.

            :param ax: The Matplotlib axes to draw the light on.
            :type ax: Matplotlib axes.
        """
        if not self.is_on:
            colour = 'gray'
        else:
            colour = self.colour
        ax.add_artist(mpatches.Circle((self.x, self.y), 0.7, color=colour))
        ax.add_artist(mpatches.Circle((self.x, self.y), 0.2, color='orange'))
        ax.plot(self.x, self.y, 'r.')

    # draw light source in a pygame display
    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        """
            A method to draw the light source in the specified PyGame display.
            A :class:`LightSource` which is switched on is drawn as a circle with colour specified by the light source's ``colour`` attribute, with a smaller circle in its center which is orange coloured.  A :class:`LightSource` which is not on will have its outer circle coloured in grey.

            :param screen: The PyGame display to draw on.
            :type screen: PyGame display

            :param scale: The scale to draw at.
            :type scale: float

            :param shiftx: The offset from centre in the x-axis for drawing.
            :type shiftx: float

            :param shifty: The offset from centre in the y-axis for drawing.
            :type shifty: float
        """
        if not self.is_on:
            colour = 'gray'
        else:
            colour = self.colour
        pygame.draw.circle(screen, center=(scale*self.x+shiftx, scale*self.y+shifty), color=colour, radius=scale*0.7)
        pygame.draw.circle(screen, center=(scale*self.x+shiftx, scale*self.y+shifty), color='orange', radius=scale*0.2)

    def get_brightness_at(self, x: float, y: float) -> float:
        """
            A method to get the brightness of the light (as it is perceived) at the given xy coordinates, according to the light source's ``model``.

            :param x: The x-component of the position to find the brightness at.
            :type x: float

            :param y: The y-component of the position to find the brightness at.
            :type y: float

            :return: The perceived brightness at the given coordinates.
            :rtype: float
        """
        brightness = 0
        if self.is_on:
            dist = self.get_distance(x, y)
            if self.model == 'inv_sq':
                brightness = self.inv_sq_model(dist)
            elif self.model == 'linear':
                brightness = self.linear_model(dist)
            elif self.model == 'binary':
                brightness = self.brightness
        return brightness

    # for some controllers, it is much easier to work with a linear light decay model. it is not physically realistic,
    # but that does not matter to this assignment
    def linear_model(self, dist: float) -> float:
        """
            A method to find the perceived brightness of the light source at the given distance, when the linear decay model is used.

            :param dist: The distance from the light source.
            :type dist: float

            :return: The perceived brightness.
            :rtype: float
        """
        return max(self.brightness - self.gradient * dist, 0)

    # this is a more realistic model of light decay. for simple Braitenberg vehicle style robots the nonlinearity of
    # light decay can lead to interesting behaviours
    def inv_sq_model(self, dist: float) -> float:
        """
            A method to find the perceived brightness of the light source at the given distance, when the inverse square decay model is used.

            :param dist: The distance from the light source.
            :type dist: float

            :return: The perceived brightness.
            :rtype: float
        """
        return self.brightness / np.power(dist+1, 2)  # 1 is added to fix brightness at dist=0

    def get_data(self) -> dict:
        """
            Get the :class:`LightSource`'s data. This method, if used, relies on the assumption that the :class:`LightSource` and its properties are static.

            :return: The :class:`LightSource`'s data in dict form.
            :rtype: dict
        """
        data = super().get_data()
        data["brightness"] = self.brightness
        data["gradient"] = self.gradient
        data["model"] = self.model
        data["colour"] = self.colour
        data["label"] = self.label
        return data

    def reset(self) -> None:
        """
            A method to reset the :class:`LightSource` to its original state.
        """
        super().reset()
        self.brightness = self.initial_brightness
        self.gradient = self.initial_gradient
        self.model = self.initial_model
        self.colour = self.initial_colour
        self.label = self.initial_label

class PerturbableLightSource(LightSource):

    def perturb(self):
        # print("Perturbable:perturb")
        self.x += random_in_interval(minimum=-1, maximum=1)
        self.y += random_in_interval(minimum=-1, maximum=1)

        self.xs[-1] = self.x
        self.ys[-1] = self.y
