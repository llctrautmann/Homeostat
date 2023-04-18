from .System import *
from .stimuli import *

# the base class for agents. currently only differential drive robots are implemented, but other types of agent could
# also be implemented easily enough
class Agent(System):
    """
        An :class:`Agent` is an abstract subclass of :class:`System`. Classes which represent specific types of mobile agents, e.g. :class:`Robot` are subclasses of :class:`Agent`.

        An :class:`Agent` is a mobile :class:`System` with position and orientation. It is expected that an :class:`Agent` will have sensors, a controller, and some way of moving through its environment.

        When you subclass :class:`Agent`, you will need to implement the following methods:

        * ``step_sensors(dt)``
        * ``control(activations, dt)``
        * ``step_actuators(speed_commands, dt)``
        * ``integrate(actual_speeds, dt)``
        * and ``update_children_positions()``
        * as well as ``pygame_draw(self, screen, scale: float, shiftx: float, shifty: float)``, if you are going to animate your simulation

        These methods split up the :class:`Agent`'s side of its sensorimotor loop. The main reason for splitting them up is to make it easier to subclass agent implementations. For example, to add sensors to an existing :class:`Agent` subclass, you would only need to override ``step_sensors`` and ``control`` - ``step_actuators`` and ``integrate`` don't need to be touched. Another example would be if you wanted to change the dynamics of motion when subclassing an existing :class:`Agent` - it may only be necessary to override the ``integrate`` method, and leave the other methods as they are.

    """

    # I'm not entirely sure about theta=None
    # - this would be an odd kind of agent!
    def __init__(self, x: float, y: float, colour: str,  theta: float=None, radius: float=1, light: LightSource=None):
        """
            __init__(x: float, y: float, colour: str,  theta: float=None, radius: float=1, light: LightSource=None)

            :param x: The :class:`Agent`'s initial x-coordinate.
            :type x: float

            :param y: The :class:`Agent`'s initial y-coordinate.
            :type y: float

            :param theta: The :class:`Agent`'s initial orientation.
            :type theta: float

            :param radius: The radius of the :class:`Agent`'s body.
            :type radius: float

            :param colour: The colour of the :class:`Agent`'s body.
            :type colour: str

            :param light: The :class:`LightSource` attached to the :class:`Agent`'s body.
            :type light: :class:`LightSource`
        """
        super().__init__(x, y, theta)  # call System constructor. xy-variables are handled there
        self.colour: str = colour
        self.radius: float = radius
        self.light: LightSource = light

    def step(self, dt: float) -> None:
        """
            Step the agent forwards in time.

            :param dt: Interval of time to integrate the noise source over - not currently used here, although it typically will be in the step methods of an :class:`Agent`'s subclasses.
            :type dt: float
        """
        # step sensors
        activations = self.step_sensors(dt)

        # step controller
        speed_commands = self.control(activations, dt)

        # step motor objects, if agent has any, or otherwise deal with any
        # dynamics of speed change such as inertia
        actual_speeds = self.step_actuators(speed_commands, dt)

        # integrate agent's motion
        self.integrate(actual_speeds, dt)

        # call System's step method
        super().step(dt)  # this call goes to System

        # update light and light sensor positions
        self.update_children_positions()

    def push(self, x: float=None, y: float=None, theta: float=None):
        """
            A method used to "push" an :class:`Agent` to a new position and orientation. The agent can be pushed in any single axis (x, y, rotation) or any combination of those axes.

            This method is here for environmental interactions such as those between an :class:`Agent` and an :class:`Arena`. The :class:`Arena` takes care of watching for collisions between agents and its walls, and when it detects one, it pushes the colliding agent back inside, using this method. It is important that this method is used, rather than just directly changing the agent's ``x``, ``y``, and ``theta`` attributes, as this method will also update the states of attached systems, such as sensors.

            :param x: The x-coordinate to push the agent to. Defaults to ``None``, in which case the agent's x-coordinate will be unchanged.
            :type x: float

            :param y: The y-coordinate to push the agent to. Defaults to ``None``, in which case the agent's y-coordinate will be unchanged.
            :type y: float

            :param theta: The orientation to push the agent to. Defaults to ``None``, in which case the agent's orientation will be unchanged.
            :type theta: float
        """
        if x:
            self.x = x
            self.xs[-1] = x
        if y:
            self.y = y
            self.ys[-1] = y
        if theta:
            self.theta = theta
            self.thetas[-1] = theta
        self.update_children_positions()
