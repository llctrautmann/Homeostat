from .base import *
from .Controller import *
from .noise import *

class BraitenbergController(Controller):
    """
        A subclass of :class:`Controller`, which can be used to control an instance of :class:`Robot`.
        It, optionally, has a noisemaker for each of the control commands to the robots two motors.
        See the ``BraitenbergController.py`` source code file for examples of controllers which implement the behaviours of simple Braitenberg Vehicles.
    """
    def __init__(self, step_fun: Callable[[float, List[float], List[float], List[float]], List[float]], left_noisemaker: NoiseSource=None, right_noisemaker: NoiseSource=None, gain=1):
        """
            __init__(step_fun: Callable[[float, List[float], List[float], List[float]], List[float]], left_noisemaker: NoiseSource=None, right_noisemaker: NoiseSource=None, gain=1)

            :param step_fun: The function which will be used to generate the controller's outputs, given the inputs to the controller's ``step`` method, the interval of time to integrate over, and any state and parameters the controller makes use of.
            :type step_fun: function

            :param left_noisemaker: A noise source which will potentially affect the command which the controller outputs for the robot's left motor.
            :type left_noisemaker: :class:`NoiseSource`

            :param right_noisemaker: A noise source which will potentially affect the command which the controller outputs for the robot's right motor.
            :type right_noisemaker: :class:`NoiseSource`

            :param gain: A gain parameter which can be used to control how fast the robot moves in response to being stimulated by a light source.
            :type gain: float
        """
        noisemakers = []
        noisemakers_inds = []
        if left_noisemaker:
            noisemakers.append(left_noisemaker)
            noisemakers_inds.append(0)

        if right_noisemaker:
            noisemakers.append(right_noisemaker)
            noisemakers_inds.append(1)

        super().__init__(inputs_n=2, commands_n=2, step_fun=step_fun, noisemakers=noisemakers, noisemakers_inds=noisemakers_inds, params=[gain])

def light_seeking(dt: float, inputs: List[float], params: List[float], state: List[float]=[]) -> List[float]:
    """
        A function, for use with a :class:`BraitenbergController`, which implements light-seeking behaviour.

        :param dt: Interval of time to integrate the controller over.
        :type dt: float

        :param inputs: The list of sensory inputs to the controller.
        :type inputs: list[float]

        :param params: The list of controller parameters (in this case, just the controller's ``gain`` attribute).
        :type params: list[float]

        :param state: This variable is included to allow elements of an agent's state to be used in its controller - in general, use of this is discouraged - where an agent's state is to be used in control, it should either be measured by an appropriate sensor, and therefore be part of the ``inputs`` vector, or it should be inferred by the controller itself.
        :type state: list[float]

        :return: List of motor commands.
        :rtype: list of floats.
    """
    # set left motor speed
    left_speed_command = params[0] * inputs[1]
    # set right motor speed
    right_speed_command = params[0] * inputs[0]

    return [left_speed_command, right_speed_command]

def light_avoiding(dt: float, inputs: List[float], params: List[float], state: List[float]=[]) -> List[float]:
    """
        A function, for use with a :class:`BraitenbergController`, which implements light-avoiding behaviour.

        :param dt: Interval of time to integrate the controller over.
        :type dt: float

        :param inputs: The list of sensory inputs to the controller.
        :type inputs: list[float]

        :param params: The list of controller parameters (in this case, just the controller's ``gain`` attribute).
        :type params: list[float]

        :param state: This variable is included to allow elements of an agent's state to be used in its controller - in general, use of this is discouraged - where an agent's state is to be used in control, it should either be measured by an appropriate sensor, and therefore be part of the ``inputs`` vector, or it should be inferred by the controller itself.
        :type state: list[float]

        :return: List of motor commands.
        :rtype: list of floats.
    """
    # set left motor speed
    left_speed_command = params[0] * inputs[0]
    # set right motor speed
    right_speed_command = params[0] * inputs[1]

    return [left_speed_command, right_speed_command]
