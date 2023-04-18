from .System import *
from .noise import *

# a class to represent a motor for a differential drive robot
# - having a class for making motor objects makes it easy to apply motor noise
# and other disturbances to the robot's motors
class Motor(System):
    """
        A class representing a motor. A motor has a maximum speed at which it will turn in either the forwards or backwards direction. Positive motor speeds results in forwards motion, and negative speeds result in backwards direction. This relationship can be reversed by setting the robot's ``reversed`` parameter to ``True``. A motor can also have inertia, which is the motor's resistance to changes of speed. If the inertia is set to 0, then the motor can change speed instantaneously. See the figure below for examples of how motors can behave with different combinations of maximum speed and inertia.

        .. figure:: images/Motor.svg
          :width: 400
          :align: center
          :alt: Motor examples

          In the figure, the response of three different motors to a constant command of 20 is shown. Motor 1 has max_speed = 10, motor_inertia_coeff = 100. Motor 2 has max_speed = 40, motor_inertia_coeff = 50. Motor 3 has max_speed = 10, motor_inertia_coeff = 0. Motor 3 changes speed to the commanded valuse in a simgle time step. The speeds of motors 1 and 3 both saturate at their maximum speed value, which is less than the commanded value. The speed of motor 2 changes most slowly, as it has the highest inertia.
    """
    # construct Motor
    # - a robot has a maximum speed. controller inputs which are larger than this
    # will saturate at the max
    # - the motor inertia coefficient determines how quickly the motor can change its speed
    #       - if the inertia is 0, then the motor can change speed instantaneously
    #           to any new control input
    #       - if the inertia is greater than 0, then the speed may change slowly,
    #       - negative inertia values will be ignored
    # - a motor can be reversed, so that forwards becomes backwards and vice versa
    def __init__(self, max_speed: float, motor_inertia_coeff: float=0, reversed: bool=False, noisemaker: NoiseSource=None):
        """
            __init__(max_speed: float, motor_inertia_coeff: float=0, reversed: bool=False, noisemaker: NoiseSource=None)


            :param max_speed: The maximum speed the motor can run at. Negative values will be converted to positive ones when they are copied to the motor's attributes.
            :type max_speed: float

            :param motor_inertia_coeff: A parameter used to determine how quickly the motor's speed can change. Defaults to 0, in which case the motor can change speed instantaneously.
            :type motor_inertia_coeff: float

            :param reversed: A parameter which determines whether the motor runs in the forwards or reverse direction. Defaults to False, in which case the motor runs forwards.
            :type reversed: bool

            :param noisemaker: A :class:`NoiseSource` object, to generate noise which is added to the motor's actual speed.
            :type noisemaker: :class:`NoiseSource`
        """
        # motors can have noise sources attached to them
        self.noisemaker = noisemaker
        # current speed and history of speed
        self.speed = 0.0
        self.speeds = [0.0]

        # system parameters
        self.motor_inertia_coeff = max(0, motor_inertia_coeff) + 1 # limits rate of change of speed
        self.initial_motor_inertia_coeff = self.motor_inertia_coeff

        self.max_speed: float = np.abs(max_speed)
        self.initial_max_speed = self.max_speed

        self.reversed = reversed
        self.reverseds = [reversed]

    # step motor forwards in time
    def step(self, speed_command: float, dt: float) -> float:
        """
            Function to step motor forward in time.

            :param speed_command: New speed command
            :type speed_command: float

            :param dt: Integration interval
            :type dt: float

            :return: Motor speed after stepping
            :rtype: float
        """
        # if motor is reversed, then reverse the control input
        if self.reversed:
            speed_command = -speed_command

        self.reverseds.append(self.reversed)

        # calculate speed change
        speed_change = (1/self.motor_inertia_coeff) * (speed_command - self.speed) # * dt

        # change speed
        self.speed += speed_change

        # apply noise
        if self.noisemaker is not None:
            self.speed += self.noisemaker.step(dt)

        # constrain motor speed
        if self.speed > 0:
            self.speed = min(self.speed, self.max_speed)
        else:
            self.speed = max(self.speed, -self.max_speed)

        # keep record of speed
        self.speeds.append(self.speed)

        # return speed
        return self.speed

    def reset(self) -> None:
        """
            A function to reset a motor to its initial state. Resets max_speed, motor_inertia_coeff, speed, history of speeds, reversed, and history of reverseds.
        """
        self.speed = self.speeds[0]
        self.speeds = [self.speed]

        self.reversed = self.reverseds[0]
        self.reverseds = [self.reversed]

        self.max_speed = self.initial_max_speed
        self.motor_inertia_coeff = self.initial_motor_inertia_coeff

        if self.noisemaker:
            self.noisemaker.reset()

    def get_data(self) -> Dict[str, Union[List[float], List[bool]]]:
        """
            A function to get a motor's data. Returns the motors histories of speed, motor direction (based on "reversed" variable), and any noise which has been applied to the motor speed.

            :return: Motor's data.
            :rtype: dict
        """
        data = {"speeds": self.speeds, "reverseds": self.reverseds, "noises": None}
        if self.noisemaker:
            data["noises"] = self.noisemaker.get_data()["noises"]
        return data
