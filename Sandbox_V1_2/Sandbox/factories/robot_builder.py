from ..base import *
from ..noise import *
from ..Robot import *
from ..BraitenbergController import *
from ..stimuli import *

import math

def light_seeking_Robot(x: float, y: float, theta: float, light_sources: List[LightSource], gain: float=1, controller: Controller=None, left_sensor_noisemaker: NoiseSource=None, right_sensor_noisemaker: NoiseSource=None, left_motor_noisemaker: NoiseSource=None, right_motor_noisemaker: NoiseSource=None, left_motor_max_speed=4, right_motor_max_speed=4, label=None, FOV=0.75*math.pi, left_sensor_angle=math.pi/4, right_sensor_angle=-math.pi/4, left_motor_inertia: float=0, right_motor_inertia: float=0) -> Robot:

    if not controller:
        controller = BraitenbergController(step_fun=light_seeking, gain=gain)

    # left_sensor_noisemaker = BrownNoiseSource(0.02)
    # right_sensor_noisemaker = BrownNoiseSource(0.01)



    # light = None
    light = LightSource(x=x, y=y, model='linear')

    left_light_sensor = LightSensor(light_sources=light_sources, x=x, y=y, noisemaker=left_sensor_noisemaker, FOV=FOV, label=label)

    right_light_sensor = LightSensor(light_sources=light_sources, x=x, y=y, noisemaker=right_sensor_noisemaker, FOV=FOV, label=label)

    sensors = [left_light_sensor, right_light_sensor]
    sensor_angles = [left_sensor_angle, right_sensor_angle]

    robbie = Robot(x=x, y=y, theta=theta, controller=controller, sensors=sensors, sensor_angles=sensor_angles, light=light, left_motor_noisemaker=left_motor_noisemaker, right_motor_noisemaker=right_motor_noisemaker,
    left_motor_max_speed=left_motor_max_speed,
    right_motor_max_speed=right_motor_max_speed, left_motor_inertia=left_motor_inertia, right_motor_inertia=right_motor_inertia)

    robbie.left_light_sensor = left_light_sensor
    robbie.right_light_sensor = right_light_sensor

    return robbie

# A subclass of Controller for an EBA robot, which takes 4 inputs due to the
# EBA_Robot having 4 sensors
# class EBAController(Controller):
#
#     # init controller with passed in noisemakers and control parameters
#     def __init__(self, left_noisemaker=None, right_noisemaker=None, gain=1):
#         # NOTE: THIS CALL TO SUPER MUST BE HERE
#         super().__init__(left_noisemaker, right_noisemaker)
#         self.gain = gain
#
#     # step method. depending on the values of speed and ratio, the robot will drive along a circular path
#     #   - but noise will be added to the control outputs, so the robot might not achieve its goal!
#     def step(self, inputs, dt):
#
#         '''
#             Set motor speeds, based on sensory input.
#
#             inputs[1] is left sensor output
#             inputs[2] is right sensor output
#             inputs[0] is second left sensor output
#             inputs[3] is second right sensor output
#
#             self.left_speed_command is the command which will be sent to the left motor
#             self.right_speed_command is the command which will be sent to the right motor
#
#         '''
#
#         # set left motor speed
#         self.left_speed_command = self.gain * inputs[2] + 0.2
#         # set right motor speed
#         self.right_speed_command = self.gain * inputs[1]
#
#         # add to left motor speed
#         self.left_speed_command += 3 * self.gain * inputs[0]
#         # add to right motor speed
#         self.right_speed_command += 2 * self.gain * inputs[3]
#
#         return super().step(inputs, dt)
#
# def an_EBA_robot(x, y, theta, light_sources):
#     FOV = 0.75 * np.pi
#
#     sensors = []
#     sensor_angles = [np.pi/5, np.pi/4, -np.pi/4, -np.pi/5]
#     for _ in range(4):
#         sensors.append(LightSensor(light_sources=light_sources, x=x, y=y, noisemaker=WhiteNoiseSource(min_val=-0.2, max_val=0.2), FOV=FOV))
#
#     sensors[0].label = 'red'
#     sensors[0].colour = 'red'
#     sensors[3].label = 'red'
#     sensors[3].colour = 'red'
#
#     sensors[1].label = 'yellow'
#     sensors[1].colour = 'yellow'
#     sensors[2].label = 'yellow'
#     sensors[2].colour = 'yellow'
#
#     robbie = Robot(x=x, y=y, theta=theta, controller=EBAController(), sensors=sensors, sensor_angles=sensor_angles, light=None)
#
#     return robbie

def light_seeking_Perturbable(x: float, y: float, theta: float, light_sources: List[LightSource], gain: float=1, controller: Controller=None, left_sensor_noisemaker: NoiseSource=None, right_sensor_noisemaker: NoiseSource=None, left_motor_noisemaker: NoiseSource=None, right_motor_noisemaker: NoiseSource=None, left_motor_max_speed=4, right_motor_max_speed=4, label=None, FOV=0.75*math.pi, left_sensor_angle=math.pi/4, right_sensor_angle=-math.pi/4, left_motor_inertia: float=0, right_motor_inertia: float=0) -> Robot:

    if not controller:
        controller = BraitenbergController(step_fun=light_seeking, gain=gain)

    # left_sensor_noisemaker = BrownNoiseSource(0.02)
    # right_sensor_noisemaker = BrownNoiseSource(0.01)



    # light = None
    light = LightSource(x=x, y=y, model='linear')

    left_light_sensor = LightSensor(light_sources=light_sources, x=x, y=y, noisemaker=left_sensor_noisemaker, FOV=FOV, label=label)

    right_light_sensor = LightSensor(light_sources=light_sources, x=x, y=y, noisemaker=right_sensor_noisemaker, FOV=FOV, label=label)

    sensors = [left_light_sensor, right_light_sensor]
    sensor_angles = [left_sensor_angle, right_sensor_angle]

    robbie = Perturbable(x=x, y=y, theta=theta, controller=controller, sensors=sensors, sensor_angles=sensor_angles, light=light, left_motor_noisemaker=left_motor_noisemaker, right_motor_noisemaker=right_motor_noisemaker,
    left_motor_max_speed=left_motor_max_speed,
    right_motor_max_speed=right_motor_max_speed, left_motor_inertia=left_motor_inertia, right_motor_inertia=right_motor_inertia)

    robbie.left_light_sensor = left_light_sensor
    robbie.right_light_sensor = right_light_sensor

    return robbie
