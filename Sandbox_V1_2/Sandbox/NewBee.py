from .base import *
from .stimuli import *
from .sensors import *
from .noise import *
from .Agent import *
from .Controller import *
from .Motor import *

from typing import Dict, List

import math

class NewBeeController(Controller):

    def __init__(self, inputs_n: int, step_fun: Callable[[List[float], float, float, float], List[float]], theta_noisemaker: NoiseSource=None, speed_noisemaker: NoiseSource=None, heading_noisemaker: NoiseSource=None):

        super().__init__(inputs_n=inputs_n, commands_n=3, step_fun=step_fun) # call System init
        # noise can be applied to the controller's outputs
        self.theta_noisemaker: NoiseSource = theta_noisemaker
        self.speed_noisemaker: NoiseSource = speed_noisemaker
        self.heading_noisemaker: NoiseSource = heading_noisemaker

        self.inputs: List[List[float]] = [[0.] * inputs_n]

        self.theta_command: float = 0.
        self.speed_command: float = 0.
        self.heading_command: float = 0.
        self.theta_commands: List[float] = [0.]
        self.speed_commands: List[float] = [0.]
        self.heading_commands: List[float] = [0.]

        self.step_fun: Callable[[List[float], float, float, float], List[float]] = step_fun

    def step(self, inputs: List[float], dt: float, theta: float, speed: float, heading: float) -> List[float]:
        # record inputs - this is primarily for potential use in the controller,
        # rather than for plotting - inputs will generally already be stored
        # elsewhere, e.g. in sensor objects
        self.inputs.append(inputs)

        self.theta_command, self.speed_command, self.heading_command = self.step_fun(inputs, dt, theta, speed, heading, self)

        if self.theta_noisemaker:
            self.theta_command += self.theta_noisemaker.step(dt)
        if self.speed_noisemaker:
            self.speed_command += self.speed_noisemaker.step(dt)
        if self.heading_noisemaker:
            self.heading_command += self.heading_noisemaker.step(dt)

        # PREVENT NEGATIVE SPEEDS?
        # self.theta_command = np.max(0, self.theta_command)

        # APPLY NOISE

        # APPLY INERTIA HERE?

        self.theta_commands.append(self.theta_command)
        self.speed_commands.append(self.speed_command)
        self.heading_commands.append(self.heading_command)

        # print(self.theta_command, self.speed_command)

        self.t += dt

        return [self.theta_command, self.speed_command, self.heading_command]

class NewBee(Agent):

    def __init__(self, x: float, y: float, theta: float, heading: float, controller: Controller, sensors: List[Sensor], sensor_angles: List[float], radius: float=1, max_speed: float=2, theta_inertia: float=0, speed_inertia: float=0, heading_inertia: float=0, colour: str='0x00FFFFFF', light=None):

        super().__init__(x, y, colour, theta, radius, light)  # call Agent constructor

        self.sensors: List[Sensor] = sensors
        self.sensor_angles: List[float] = sensor_angles
        self.initial_sensor_angles: List[float] = []
        # I CAN JUST COPY THE LIST DIRECTLY!
        for sensor_angle in sensor_angles:
            self.initial_sensor_angles.append(sensor_angle)

        self.max_speed: float = max_speed
        self.theta_inertia: float = theta_inertia
        self.speed_inertia: float = 0

        self.controller: NewBeeController = controller

        self.heading = heading
        self.headings: List[float] = [heading]

        self.speed: float = 0
        self.speeds: List[float] = [self.speed]

        self.speed_motor = Motor(max_speed=max_speed, motor_inertia_coeff=0, reversed=False, noisemaker=None)
        self.theta_motor = Motor(max_speed=1, motor_inertia_coeff=10, reversed=False, noisemaker=None)
        self.heading_motor = Motor(max_speed=1, motor_inertia_coeff=10, reversed=False, noisemaker=None)

        self.update_children_positions()

    # update positions and orientations of all sensors
    def update_children_positions(self) -> None:

        # update light positions
        if self.light:
            self.light.x = self.x
            self.light.y = self.y

        # update sensor positions
        for i, sensor in enumerate(self.sensors):
            sensor.x = self.x + (self.radius * np.cos(self.theta + self.sensor_angles[i]))
            sensor.y = self.y + (self.radius * np.sin(self.theta + self.sensor_angles[i]))
            sensor.theta = self.thetas[-1] + self.sensor_angles[i]

    def step_sensors(self, dt: float) -> List[float]:
        activations: List[float] = []
        for sensor in self.sensors:
            s = sensor.step(dt)
            # print(s)
            # print(sensor.light_sources)
            # activations.append(sensor.step(dt))
            activations.append(s)

        return activations

    def integrate(self, speeds: List[float], dt: float) -> None:

        self.speed = speeds[1]
        self.speeds.append(self.speed)

        self.theta += speeds[0]

        self.heading += speeds[2]
        self.headings.append(self.heading)

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def control(self, activations: List[float], dt: float) -> List[float]:

        return self.controller.step(activations, dt, self.theta, self.speed, self.heading)

    def step_actuators(self, speed_commands: List[float], dt: float) -> List[float]:

        # speed inertia
        speed_commands[1] = self.speed_motor.step(dt=dt, speed_command=speed_commands[1])
        # body rotation - the way the Bee is looking
        speed_commands[0] = self.theta_motor.step(dt=dt, speed_command=speed_commands[0])
        # heading - the way the Bee is flying
        speed_commands[2] = self.heading_motor.step(dt=dt, speed_command=speed_commands[2])

        return speed_commands

    # draw bee in the specified matplotlib axes
    def draw(self, ax) -> None:
        '''
            Draw bee in specified Matplotlib axes.
        '''
        ax.plot([self.x, self.x+self.radius*np.cos(self.theta)],
                 [self.y, self.y+self.radius*np.sin(self.theta)], 'k--', linewidth='2')
        ax.add_artist(mpatches.Circle((self.x, self.y), self.radius, color='blue'))

        for sensor in self.sensors:
            sensor.draw(ax)
            # self.__draw_FOV(sensor, ax)

        if self.light:
            self.light.draw(ax)

    # draw robot in a pygame display
    def pygame_draw(self, screen, scale: float, shiftx: float, shifty: float) -> None:

        '''



        '''

        self.__pygame_draw_wings(screen, scale, shiftx, shifty)

        pygame.draw.circle(screen, center=(scale*self.x+shiftx, scale*self.y+shifty), color=self.colour, radius=scale*self.radius)

        for sensor in self.sensors:
            sensor.pygame_draw(screen, scale, shiftx, shifty)

        end_x = self.x + self.radius * np.cos(self.theta)
        end_y = self.y + self.radius * np.sin(self.theta)
        pygame.draw.line(screen, color='green',
                         start_pos=(scale * self.x + shiftx, scale * self.y + shifty),
                         end_pos=(scale * end_x + shiftx, scale * end_y + shifty), width=2)

        if self.light:
            self.light.pygame_draw(screen, scale, shiftx, shifty)

        end_x = self.x + self.radius * np.cos(self.heading) * 1.3
        end_y = self.y + self.radius * np.sin(self.heading) * 1.3
        pygame.draw.line(screen, color='red',
                         start_pos=(scale * self.x + shiftx, scale * self.y + shifty),
                         end_pos=(scale * end_x + shiftx, scale * end_y + shifty), width=2)

    def __pygame_draw_wings(self, screen, scale: float, shiftx: float, shifty: float) -> None:
        a1 = 3*math.pi/8
        a_inc = 0.35
        offset = 0.2
        l = 1.6
        self.__pygame_draw_wing(screen, scale, shiftx, shifty, a1, a1+a_inc, 7, offset, l)

        self.__pygame_draw_wing(screen, scale, shiftx, shifty, -a1, -(a1+a_inc), 7, offset, l)

        a1 = 5*math.pi/8
        a_inc = 0.25
        offset = 0.2
        l = 1.6

        self.__pygame_draw_wing(screen, scale, shiftx, shifty, a1, a1+a_inc, 5, offset, l)

        self.__pygame_draw_wing(screen, scale, shiftx, shifty, -a1, -(a1+a_inc), 5, offset, l)

    def __pygame_draw_wing(self, screen, scale: float, shiftx: float, shifty: float, start_angle: float, end_angle:float, n: int, offset: float, l: float) -> None:

        angles = np.linspace(start_angle, end_angle, n)
        for angle in angles:
            x1, y1, x2, y2 = self.wing_ends(offset, angle, l)

            pygame.draw.line(screen, color='red',
                             start_pos=(scale * x1 + shiftx, scale * y1 + shifty),
                             end_pos=(scale * x2 + shiftx, scale * y2 + shifty), width=2)

    def wing_ends(self, offset_scale, angle, relative_length):

        a = angle + self.theta

        offset = offset_scale * self.radius
        x1 = self.x + offset * math.cos(self.theta)
        y1 = self.y + offset * math.sin(self.theta)

        x2 = x1 + relative_length * math.cos(a)
        y2 = y1 + relative_length * math.sin(a)

        return x1, y1, x2, y2

def circler(dt, inputs, theta, speed, heading, controller):

    return [0, 2, 0.01]

def wyrder(dt, inputs, theta, speed, heading, controller):

    return [0. * math.sin(controller.t + 1), 20 * math.sin(controller.t), 0.01]
