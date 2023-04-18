from .base import *
from .stimuli import *
from .sensors import *
from .Agent import *
from .Motor import *
from .Controller import *
from .noise import *
from .pygame_functions import *

"""
.. module:: models
   :synopsis: Contains model of a Teacher Record
.. moduleauthor:: Rich Yap <github.com/richyap13>
"""

class Robot(Agent):
    """
        A :class:`Robot` is a subclass of :class:`Agent`, and represents a mobile robot with a differential drive, i.e. two motors, on either side of its body, which can be driven at different speeds. A :class:`Robot` can have an arbitrary number of :class:`LightSensor` objects. It can also have a :class:`LightSource` attached to its body, so that it can be seen by other agents with light sensors.


        .. figure:: images/Robot.png
          :width: 600
          :align: center
          :alt: Robot, as drawn in Matplotlib

          A :class:`Robot`, as drawn in Matplotlib. While this example has two light sensors, much like Braitenberg's simpler vehicles, a :class:`Robot` can have an arbitrary number of sensors.
    """

    ## Documentation for a method.
    # @type controller: Controller
    # @param controller: The robot's Controller


    def __init__(self, x: float, y: float, controller: Controller, sensors: List[LightSensor], sensor_angles: List[float], radius: float=1, theta: float=0, left_motor_max_speed: float=2, right_motor_max_speed: float=2, left_motor_inertia: float=0, right_motor_inertia: float=0, left_motor_noisemaker: NoiseSource=None, right_motor_noisemaker: NoiseSource=None, left_motor_reversed: bool=False, right_motor_reversed: bool=False, colour: str='darkblue', light: LightSource=None):
        """
            __init__(x: float, y: float, controller: Controller, sensors: List[LightSensor], sensor_angles: List[float], radius: float=1, theta: float=0, left_motor_max_speed: float=2, right_motor_max_speed: float=2, left_motor_inertia: float=0, right_motor_inertia: float=0, left_motor_noisemaker: NoiseSource=None, right_motor_noisemaker: NoiseSource=None, left_motor_reversed: bool=False, right_motor_reversed: bool=False, colour: str='darkblue', light: LightSource=None)

            :param x: Robot's initial x-coordinate
            :type x: float

            :param y: Robot's initial y-coordinate
            :type y: float

            :param theta: Robot's initial orientation, in radians. Defaults to ``0.0``
            :type theta: float

            :param radius: Radius of robot's body. Defaults to ``1.0``
            :type radius: float

            :param controller: Robot's controller system
            :type controller: Instance of subclass of :class:`Controller`

            :param sensors: List of robot's sensor systems
            :type sensors: List of instances of subclasses of :class:`Sensor` The subclass will typically be :class:`LightSensor` , but it can be others as long as their step methods have the same inputs/output

            :param sensor_angles: List of the angles corresponding to the list of sensors
            :type sensor_angles: List of floats

            :param left_motor_max_speed: Maximum speed (magnitude) of robot's left motor (which is an instance of :class:`Motor`). Negative values will be converted to positive ones when the motor is initialised.
            :type left_motor_max_speed: float

            :param right_motor_max_speed: Maximum speed (magnitude) of robot's right motor (which is an instance of :class:`Motor`). Negative values will be converted to positive ones when the motor is initialised.
            :type right_motor_max_speed: float

            :param left_motor_inertia: Inertia parameter for the robot's left motor
            :type left_motor_inertia: float

            :param right_motor_inertia: Inertia parameter for the robot's right motor
            :type right_motor_inertia: float

            :param left_motor_noisemaker: Noise source for robot's left motor (usually an instance of :class:`NoiseSource`). Defaults to ``None``
            :type left_motor_noisemaker: NoiseSource

            :param right_motor_noisemaker: Noise source for robot's right motor (usually an instance of :class:`NoiseSource`). Defaults to ``None``
            :type right_motor_noisemaker: NoiseSource

            :param left_motor_reversed: A flag which is used to detrmine whether or not the robot's left motor runs in the reverse direction. Defaults to ``False``
            :type left_motor_reversed: bool

            :param right_motor_reversed: A flag which is used to detrmine whether or not the robot's right motor runs in the reverse direction. Defaults to ``False``
            :type right_motor_reversed: bool

            :param colour: Colour of the robot's body. Defaults to ``'darkblue'``
            :type colour: str

            :param light: The light which is attached to the robot, so that it can be detected by other robots (instance of :class:`LightSource`). Defaults to ``None``, in which case the robot has no light attached.
            :type light: Instance of :class:`Light`

            |

            .. only:: comment

                This vertical bar forces a new line

        """
        super().__init__(x, y, colour, theta, radius, light)  # call Agent constructor

        self.controller: Controller = controller  # the controller for the robot, which will set motor speeds according to how stimulated the robot's sensors are

        self.sensors: List[LightSensor] = sensors
        self.initial_sensors = sensors
        self.sensor_angles: List[float] = sensor_angles
        self.initial_sensor_angles: List[float] = []
        # COULD DO STRAIGHT COPY OF LIST HERE
        for sensor_angle in sensor_angles:
            self.initial_sensor_angles.append(sensor_angle)
        self.update_children_positions()  # update sensor positions according to robot's state

        self.left_motor: Motor = Motor(max_speed=left_motor_max_speed, motor_inertia_coeff=left_motor_inertia, reversed=left_motor_reversed, noisemaker=left_motor_noisemaker)
        self.right_motor: Motor = Motor(max_speed=right_motor_max_speed, motor_inertia_coeff=right_motor_inertia, reversed=right_motor_reversed, noisemaker=right_motor_noisemaker)

    # update positions and orientations of all sensors
    def update_children_positions(self) -> None:
        """
            This method is used to update the positions and orientations of a robot's attached subsystems, such as its sensors, as the robot moves.
        """
        # update light positions
        if self.light:
            self.light.x = self.x
            self.light.y = self.y

        # update sensor positions
        for i, sensor in enumerate(self.sensors):
            sensor.x = self.x + (self.radius * np.cos(self.theta + self.sensor_angles[i]))
            sensor.y = self.y + (self.radius * np.sin(self.theta + self.sensor_angles[i]))
            sensor.theta = self.thetas[-1] + self.sensor_angles[i]

    def reset(self, reset_controller: bool=True) -> None:
        """
            This method resets a robot's state and simulation data to their initial values, so that it can be used again.

            :param reset_controller: determines whether or not the robot's controller is also reset, defaults to ``True``. This is because sometimes you might want to reset a robot and simulate it again taking advantage of any information or learning which the controller has acquired.
            :type reset_controller: bool
        """
        super().reset()
        self.left_motor.reset()
        self.right_motor.reset()

        # this assumes that no sensors have been added or removed
        for i, sensor in enumerate(self.sensors):
            sensor.reset()
            self.sensor_angles[i] = self.initial_sensor_angles[i]
            self.sensors[i] = self.initial_sensors[i]
        self.update_children_positions()

        if reset_controller:
            self.controller.reset()

    def step_actuators(self, speed_commands: List[float], dt: float) -> List[float]:
        """

        """
        left_speed = self.left_motor.step(speed_commands[0], dt)
        right_speed = self.right_motor.step(speed_commands[1], dt)

        return [left_speed, right_speed]

    # this is separated from the step method as it is easier to override in any subclasses of Robot than step, which
    # should be the same for all Robots
    # - the reason we would do this is if we wanted to change the number or kinds of input
    #   which the Robot would have (e.g. because we want to use more than two sensors)
    def control(self, activations: List[float], dt: float) -> List[float]:
        """

            Only called from step().

            A method which gets motor speed commands by calling the step method of the robot's controller.

        """
        # get motor speeds from controller
        left_speed, right_speed = self.controller.step(dt, activations)

        # return speeds to step method
        return [left_speed, right_speed]

    # this is separated from the step method in case we want to override it
    # - one example of why we might want to do this is if we wanted to add collisions
    #   to the simulation. to achieve this, we could do something like create a subclass of
    #   Robot, with its own integrate method, which calls this one and then superimposes
    #   an additional movement due to collisions
    def integrate(self, speeds: List[float], dt: float) -> None:
        """

            Only called from step().

            Applies a motor activation vector to an agent state, and simulates the consequences using Euler integration over a dt interval.

        """
        # calculate the linear speed and angular speed
        v = np.mean([speeds[0], speeds[1]])
        omega = (speeds[1] - speeds[0]) / (2.0 * self.radius)

        state = np.array([self.x, self.y, self.theta])

        # calculate time derivative of state
        deriv = [v * np.cos(state[2]), v * np.sin(state[2]), omega]

        # perform Euler integration
        state = dt * np.array(deriv) + state

        # store robot state
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]

    def step_sensors(self, dt: float) -> List[float]:
        '''
            Only called from step().

            A method which steps the sensors in the robot's `sensors` list, and returns the sensor activations in a list.
        '''
        activations = []
        for sensor in self.sensors:
            s = sensor.step(dt)
            # print(s)
            # print(sensor.light_sources)
            # activations.append(sensor.step(dt))
            activations.append(s)

        return activations

    def get_data(self) -> Dict[str, Dict[str, Any]]:
        '''
            Get the robot's simulation data, including the data from its sensors, motors and controller.
        '''
        data = super().get_data()

        data["classname"] = "Robot"

        data["sensors"] = []
        for sensor in self.sensors:
            data["sensors"].append(sensor.get_data())

        data["left_motor"] = self.left_motor.get_data()
        data["right_motor"] = self.right_motor.get_data()

        data["controller"] = self.controller.get_data()

        return data

    # draw robot in the specified matplotlib axes
    def draw(self, ax) -> None:
        '''
            Draw robot in specified Matplotlib axes.
        '''
        ax.plot([self.x, self.x+self.radius*np.cos(self.theta)],
                 [self.y, self.y+self.radius*np.sin(self.theta)], 'k--', linewidth='2')
        ax.add_artist(mpatches.Circle((self.x, self.y), self.radius, color=self.colour))
        wheels = [mpatches.Rectangle((-0.5*self.radius, y), width=self.radius, height=0.2*self.radius, color="black") for y in (-1.1*self.radius, 0.9*self.radius)]
        tr = mtransforms.Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
        for wheel in wheels:
            wheel.set_transform(tr)
            ax.add_artist(wheel)

        for sensor in self.sensors:
            sensor.draw(ax)
            # self.__draw_FOV(sensor, ax)

        if self.light:
            self.light.draw(ax)

    def __wheel_ends(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """

        """
        offset = 0.95

        left_mid_x = self.x + (offset * self.radius * math.cos(self.theta + math.pi/2))
        left_mid_y = self.y + (offset * self.radius * math.sin(self.theta + math.pi/2))

        right_mid_x = self.x + (offset * self.radius * math.cos(self.theta + -math.pi/2))
        right_mid_y = self.y + (offset * self.radius * math.sin(self.theta + -math.pi/2))

        half_wheel_len = 0.5 * self.radius

        left_end_x = left_mid_x + half_wheel_len * math.cos(self.theta)
        left_end_y = left_mid_y + half_wheel_len * math.sin(self.theta)

        right_end_x = right_mid_x + half_wheel_len * math.cos(self.theta)
        right_end_y = right_mid_y + half_wheel_len * math.sin(self.theta)

        left_end_x2 = left_mid_x - half_wheel_len * math.cos(self.theta)
        left_end_y2 = left_mid_y - half_wheel_len * math.sin(self.theta)

        right_end_x2 = right_mid_x - half_wheel_len * math.cos(self.theta)
        right_end_y2 = right_mid_y - half_wheel_len * math.sin(self.theta)

        return left_end_x, left_end_y, right_end_x, right_end_y, left_end_x2, left_end_y2, right_end_x2, right_end_y2

    # draw robot in a pygame display
    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        '''
            Draw robot on PyGame screen.
        '''
        self.__pygame_draw_wheels(screen, scale, shiftx, shifty)

        pygame_drawcircle(screen, shiftx, shifty, scale, self.x, self.y, self.radius, self.colour)

        for sensor in self.sensors:
            sensor.pygame_draw(screen, scale, shiftx, shifty)

        end_x = self.x + self.radius * np.cos(self.theta)
        end_y = self.y + self.radius * np.sin(self.theta)
        pygame_drawline(screen, shiftx, shifty, scale, self.x, self.y, end_x, end_y, 'green', 2)

        if self.light:
            self.light.pygame_draw(screen, scale, shiftx, shifty)

    def __pygame_draw_wheels(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        """

        """
        left_end_x, left_end_y, right_end_x, right_end_y, left_end_x2, left_end_y2, right_end_x2, right_end_y2 = self.__wheel_ends()

        pygame_drawline(screen, shiftx, shifty, scale, left_end_x, left_end_y, left_end_x2, left_end_y2, 'red', 6)

        pygame_drawline(screen, shiftx, shifty, scale, right_end_x, right_end_y, right_end_x2, right_end_y2, 'red', 6)

class Perturbable(Robot):

    def perturb(self):
        # print("Perturbable:perturb")
        self.x += random_in_interval(minimum=-10, maximum=10)
        self.y += random_in_interval(minimum=-10, maximum=10)
        self.theta += random_in_interval(minimum=-1, maximum=1)

        self.xs[-1] = self.x
        self.ys[-1] = self.y
        self.thetas[-1] = self.theta
