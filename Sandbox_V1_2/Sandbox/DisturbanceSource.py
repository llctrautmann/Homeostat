from .System import *
from .noise import *
from .Robot import *
from .stimuli import *
import copy as cp

from typing import List

# base class for sources of disturbance. disturbances can be easily set up to be enabled and disabled at predetermined
# times during a run of a simulation
#   at every time in start_times, the disturbance should become active
#   at every time in stop_times, the disturbance should be disabled again
#   N.B. some of these disturbances are severe enough to require adaptation in a robot's controller, and some might be
#   easier to deal with than others
#   - using more than one of these at once could be very challenging
#   - they are fairly easy to create, if you have ideas of your own for disturbances
class DisturbanceSource(System):
    """
        An abstract superclass for timed sources of disturbances. A :class:`DisturbanceSource` is a subclass of :class:`System`, which has no position.

        Subclasses of :class:`DisturbanceSource` must call its ``step()`` method using ``super().step(dt)``, so that disturbances are enabled and disabled at the right times.

        When the :class:`DisturbanceSource` is constructed, it is passed lists of times at which to be activated (enabled) and deactivated (disabled). A :class:`DisturbanceSource` is disabled by default when first created. When a :class:`DisturbanceSource` is active, the list of times to disable it is monitored, and when it is not active, the list of times to enable it is monitored. When the :class:`DistubanceSource`'s internal clock, ``t``, exceeds the first time in the currently monitored list, then the :class:`DisturbanceSource` will be enabled/disabled as appropriate. When this happens, the listed time which has just been used will be removed from the list, so that the next time in the list (if there is one) will become the first element in the list.

        As it is always the first elements in the two lists which are monitored, the two together must be ordered correctly with respect to time. For example, if ``start_times = [100, 200]`` and ``end_times = [50, 130]``, then the disturbance will be activated at t >= 100, and then deactivated in the next simulation step, as then t > 100 > 50. Then the same thing will happen again, at t >= 200 and in the following simulation step, because 200 > 130.

        If no end_times are specified, then once any disturbance is active, it will be so until the simulation ends. If you have more start times than end times, and they are appropriately sequenced, then there will be a time at which the disturbance will be activated without ever being deactivated again. For example, if ``start_times = [100, 200, 500]`` and ``end_times = [150, 300]``, then the disturbance will be enabled and disabled twice, and then be enabled from t = 500 until the simulation ends (assuming that the simulation duration > 500).

        If you would like a disturbance to be 'one-shot', i.e. only active for a single simulation step, then the easiest way to implement that is in the step() method of the subclass of :class:`DisturbanceSource`, e.g. as in the :class:`SensoryInversionDisturbanceSource` class ``step()`` method, where whenever the disturbance is applied, ``self.enabled`` is subsequently set to ``False`` to disable it again.
    """
    # construct disturbance source
    def __init__(self, start_times: List[int]=[], stop_times: List[int]=[], enabled: bool=True):
        """
            __init__(start_times: List[int]=[], stop_times: List[int]=[], enabled: bool=True)

            :param start_times: The list of times at which a :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param stop_times: The list of times at which a :class:`DisturbanceSource` is deactivated.
            :type stop_times: list[int]

            :param enabled: A parameter which determines whether or not a :class:`DisturbanceSource` is active. The :class:`DisturbanceSource` manages this variable itself. In most cases, it will be best to leave this variable as ``False`` when a :class:`DisturbanceSource` is constructed, and only use ``start_times`` and ``stop_times`` to inform the disturbance source when to change it.
            :type enabled: bool

        """
        super().__init__()
        self.enabled = enabled
        if start_times:  # if any start_times are passed in, then the disturbance is initially disabled, and it will not be enabled until the first start time is reached
            self.enabled = False
        self.t: float = 0.0  # DisturbanceSources keep track of time internally. a number of classes do this, and it is certainly not the most computationally efficient approach, but it avoids using a global time variable (which could be horrendous)
        self.start_times = start_times
        self.stop_times = stop_times
        self.init_enabled = self.enabled

        # I CAN DIRECTLY COPY THESE LISTS?
        self.init_start_times = []
        for s in start_times:
            self.init_start_times.append(s)

        self.init_stop_times = []
        for s in stop_times:
            self.init_stop_times.append(s)

    def step(self, dt: float) -> None:
        """
            Step the disturbance source forwards in time. This is where the disturbance gets enabled and disabled. The effect of the disturbance is be implemented in the step methods for the subclasses of DisturbancSource

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        self.t += dt  # increment time variable by simulatino step size
        if self.enabled:
            if not self.stop_times:  # if the disturbance is enabled, then check if stop_times is empty, and if it is, then there is nothing left to do, so just return
                return
            else:  # otherwise, check to see if it is time to disable the disturbance yet
                if self.t > self.stop_times[0]:  # always get the end_time at the beginning of the list
                    self.enabled = False
                    self.stop_times.pop(0)  # every time an end time is used, it is removed (popped) from the list as it will not be used again
        else:
            if not self.start_times:  # if the disturbance is disabled but there are no start_times remaining, then there is nothing left to do, so just return
                return
            else:  # otherwise, check to see if it is time to enable the disturbance yet
                if self.t > self.start_times[0]:  # always get the start_time at the beginning of the list
                    self.enabled = True
                    self.start_times.pop(0)  # every time a start time is used, it is removed (popped) from the list as it will not be used again

    def reset(self) -> None:
        """
            Reset DisturbanceSource by resetting its timer, enabled state and start/stop time lists.
        """
        self.enabled = self.init_enabled
        self.t = 0
        self.start_times = []
        self.stop_times = []

        for s in self.init_start_times:
            self.start_times.append(s)

        for s in self.init_stop_times:
            self.stop_times.append(s)


# it uses a WhiteNoiseSource, which is effectively integrated over time to cause
# drift in the robot's sensor positions
#   this is a gradual change to the robot's morphology which, if big enough, will
#   make it necessary for the controller to adapt
#   depending on how large the maximum move is, and how long the disturbance is
#   applied for, this disturbance can be a major problem, and it may not even be
#   possible for most controllers to adapt if, for example, both sensors end up
#   on the same side of the robot's body
class MovingSensorsDisturbanceSource(DisturbanceSource):
    """
        A class for making a :class:`Robot`'s light sensors gradually change, through
        small random movements, their angular position on the robot's body. The
        sensors will always be the body's perimeter, and change their orientations to
        match their new angular positions.
    """
    # construct disturbance source
    def __init__(self, robot: Robot, max_move: float=np.pi/36, move_left: float=True, move_right: float=True, start_times: List[int]=[], stop_times: List[int]=[], enabled: bool=True, sensor_indices: List[int]=[0, 1]):
        """
            __init__(robot, max_move=np.pi/36, move_left=True, move_right=True, start_times=[], stop_times=[], enabled=True, sensor_indices=[0, 1])

            :param robot: The class:`Robot` instance which will be disturbed.
            :type robot: class:`Robot`

            :param max_move: The maximum angle by which a sensor will move in a single simulation step.
            :type max_move: float

            :param move_left: Not used in this implementation.
            :type move_left: bool

            :param move_right: Not used in this implementation.
            :type move_right: bool

            :param start_times: The list of times at which a :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param stop_times: The list of times at which a :class:`DisturbanceSource` is deactivated.
            :type stop_times: list[int]

            :param enabled: A parameter which determines whether or not a :class:`DisturbanceSource` is active.

            :param sensor_indices: The indices of the sensors, in the robot's sensor list, which will be disturbed.
            :type sensor_indices: list[int]
        """
        super().__init__(start_times, stop_times, enabled)
        self.robot = robot
        self.move_left = move_left  # left sensor will be affected, if this is True
        self.move_right = move_right  # right sensor will be affectd, if this is True
        self.noisesource = WhiteNoiseSource(min_val=-max_move, max_val=max_move)
        self.sensor_indices = sensor_indices

    # step disturbance
    def step(self, dt: float) -> None:
        """
            Step :class:`MovingSensorsDisturbanceSource` forwards in time. The sensors specified in ``sensor_indices`` will be moved by a random amount whenever the disturbance source is enabled.

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        super().step(dt)
        if self.enabled:
            for i in self.sensor_indices:
                self.robot.sensor_angles[i] += self.noisesource.step(dt)

    def reset(self) -> None:
        """
            Reset DisturbanceSource by resetting its timer, enabled state and start/stop time lists, as well as its noise source.
        """
        super().reset()
        self.noisesource.reset()

# this implements a large-scale sensory inversion, which will be quite easy to adapt to, but only if it can be reliably
# detected.
#   - in the case of this disturbance, detection may be the hard part
#   - a relatively easy place to begin would be with a light-seeking robot in an environment with only one light source
#       - then it shouldn't be too hard to detect the sudden change in behaviour this disturbance will cause
#   - in more complex environments, and/or if combined with other disturbances and noise, this becomes a real challenge
class SensoryInversionDisturbanceSource(DisturbanceSource):
    """
        A class for switching the positions of two of a :class:`Robot`'s sensors.

        An example of how this will often be used is in a :class:`Robot` with two
        :class:`LightSensor` s, like Braitenberg's light-seeking and light-avoiding vehicles. Braitenberg's *aggressor* (light-seeker) can be turned into his *coward* (light-avoider) simply by changing its sensorimotor connections from crossed to direct. Keeping the connections the same and swapping the positions of the sensors has the same effect.

        For a :class:`Robot` with more than two sensors, this disturbance source can be used with any pair, specified by ``sensor_indices`` when the :class:`SensoryInversionDisturbanceSource` is constructed.
    """
    # construct disturbance source
    def __init__(self, robot: Robot, start_times: List[int], sensor_indices: List[int]=[0, 1]):
        """
            __init__(robot, start_times, sensor_indices=[0, 1])

            :param robot: The :class:`Robot` which will be disturbed.
            :type robot: :class:`Robot`

            :param start_times: The list of times at which a :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param sensor_indices: The indices of the two sensors which will have their positions switched.
            :type sensor_indices: list[int]
        """
        super().__init__(start_times, [], False)
        self.robot = robot  # the robot which will be disturbed
        self.sensor_indices = sensor_indices
        self.initial_angles = self.robot.sensor_angles

    # step disturbance source
    def step(self, dt: float) -> None:
        """
            Step :class:`SensoryInversionDisturbanceSource` forwards in time. When the disturbance source becomes enabled, the sensors will be switched, and then the disturbance source will imeediately be disabled again.

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        super().step(dt)
        if self.enabled:  # rewire connections between sensor and motors
            # swap sensor angles
            temp = self.robot.sensor_angles[self.sensor_indices[0]]
            self.robot.sensor_angles[self.sensor_indices[0]] = self.robot.sensor_angles[self.sensor_indices[1]]
            self.robot.sensor_angles[self.sensor_indices[1]] = temp

            self.enabled = False  # unlike a generic DisturbanceSource, this is a one-shot disturbance, and so is automatically disable immediately after being applied

    def reset(self) -> None:
        """
            Reset DisturbanceSource by resetting its timer, enabled state and start/stop time lists, as well as robot sensor angles.
        """
        super().reset()
        self.robot.sensor_angles = self.initial_angles

class LightSwitcherDisturbanceSource(DisturbanceSource):
    """


        The light sources which will be disturbed should have both of their ``colour`` and ``label`` parameters set to either 'red' or 'yellow'.
    """
    def __init__(self, light_sources: List[LightSource], start_times: List[int], switch_colours: List[str]=['red', 'yellow']):
        """
            __init__(light_sources: List[LightSource], start_times: List[int], switch_colours: List[str]=['red', 'yellow'])

            :param light_sources: The list of light sources which will be disturbed.
            :type light_sources: list[:class:`LightSource`]

            :param start_times: The list of times at which a :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param switch_colours: The two colours which the light sources will be switched between.
            :type switch_colours: list[str]
        """
        super().__init__(start_times, [], False)

        self.light_sources = light_sources
        self.switch_colours = switch_colours

    def step(self, dt: float) -> None:
        """
            Step :class:`LightSwitcherDisturbanceSource` forwards in time. When the disturbance source becomes enabled, the lights will be switched, and then the disturbance source will imeediately be disabled again.

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        super().step(dt)
        if self.enabled:
            for s in self.light_sources:
                if s.label == 'red':
                    s.label = 'yellow'
                    s.colour = 'yellow'
                elif s.label == 'yellow':
                    s.label = 'red'
                    s.colour = 'red'

        self.enabled = False  # unlike a generic DisturbanceSource, this is a one-shot disturbance, and so is automatically disabled immediately after being applied

        def reset(self) -> None:
            """
                Reset DisturbanceSource by resetting its timer, enabled state and start/stop time lists, as well as restoring lights to their riginal states.
            """
            super().reset()

            for s in self.light_sources:
                s.reset()

class SensorLabelSwitcherDisturbance(DisturbanceSource):

    '''
        A "one-shot" disturbance source which changes the labels of the disturbed :class:`Robot`'s sensors, so that the :class:`LightSource` s which they detect are changed, according to the list of labels passed to the :class:`DisturbanceSource`'s constructor.

        Similar in effect to :class:`LightSwitcherDisturbanceSource`, except that more than two labels can be used here. For example, if the ``labels`` list is ``['yellow', 'red', 'food']``, the sensors' labels will cycle through ``'yellow'``, ``'red'``, ``'food'``, ``'yellow'``, and so on.
    '''

    def __init__(self, robot: Robot, start_times: List[int], labels: List[str]=['yellow','red'], enabled: bool=True):
        """
            __init__(robot: Robot, start_times: List[int], labels: List[str]=['yellow','red'], enabled: bool=True)

            :param robot: The :class:`Robot` to be disturbed.
            :type robot: :class:`Robot`

            :param start_times: The list of times at which a :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param stop_times: The list of times at which a :class:`DisturbanceSource` is deactivated.
            :type stop_times: list[int]

            :param enabled: A parameter which determines whether or not a :class:`DisturbanceSource` is active.

        """
        super().__init__(start_times=start_times, stop_times=[], enabled=enabled)
        self.robot = robot
        self.labels = labels
        self.labels_ind: int = 0
        self.labels_n = len(labels)

    def step(self, dt: float) -> None:
        """
            Step :class:`SensorLabelSwitcherDisturbance` forwards in time. When the disturbance source becomes enabled, the lights which the sensor detects will be switched to the next in the list of labels (which will be the first in the list, if the end has already been reached), and then the disturbance source will immediately be disabled again.

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        super().step(dt)

        if self.enabled:

            # print('i\'m disturbed at ', self.t)

            for s in self.robot.sensors:
                s.label = self.labels[self.labels_ind % self.labels_n]

            self.labels_ind += 1

            self.enabled = False

    def reset(self) -> None:
        """
            Reset disturbance, so it can be re-used.
        """
        super().reset()

        self.labels_ind = 0

class ParameterDisturbanceSource(DisturbanceSource):
    """
        A subclass of :class:`DisturbanceSource` which disturbs the parameters of a :class:`Controller`, moving them by some random distance in every simulation step when the disturbance is active.
    """
    def __init__(self, controller: Controller, start_times: List[int], enabled: bool=True):
        """
            __init__(controller: Controller, start_times: List[int], enabled: bool=True)

            :param controller: The :class:`Controller` to be disturbed.
            :type controller: :class:`Controller`

            :param start_times: The list of times at which the :class:`DisturbanceSource` is activated.
            :type start_time: list[int]

            :param stop_times: The list of times at which the :class:`DisturbanceSource` is deactivated.
            :type stop_times: list[int]

            :param enabled: The parameter which determines whether or not the :class:`DisturbanceSource` is active.
        """
        super().__init__(start_times=start_times, stop_times=[], enabled=enabled)

        self.controller = controller

        self.initial_params = cp.deepcopy(controller.params)

        self.noisesource = WhiteNoiseSource(min_val=-2, max_val=2)

    def step(self, dt) -> None:
        """
            Step :class:`ParameterDisturbanceSource` forwards in time. The disturbed controller's parameters will be disturbed by a random amount, drawn from a uniform distirbution on the interval [-2, 2], whenever the disturbance source is enabled.

            :param dt: Interval of time to integrate the disturbance source over.
            :type dt: float
        """
        super().step(dt)

        if self.enabled:
            # print(self.controller.params)
            for i in range(len(self.controller.params)):

                self.controller.params[i] = self.noisesource.step(dt)
            # print(self.controller.params)
            self.enabled = False

    def reset(self) -> None:
        """
            Reset disturbance, so it can be re-used.
        """
        super().reset()

        self.controller.params = self.initial_params
