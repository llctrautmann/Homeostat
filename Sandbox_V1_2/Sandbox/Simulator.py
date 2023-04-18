from .base import *
from .Agent import *
from .DisturbanceSource import *

from typing import List, Dict

class Simulator:
    """
        A class for running a *Sandbox* simulation.

        It is possible to write your own simulation loop, without too much difficulty, so this class really exists for convenience, for standardising, and to cut down on the amount of code in main scripts.
    """
    def __init__(self, agents: List[Agent], envs: List[System], duration: float, dt: float, obj_fun=None, disturbances: List[DisturbanceSource]=[]):
        """
            __init__(agents: List[Agent], envs: List[System], duration: float, dt: float, obj_fun=None, disturbances: List[DisturbanceSource]=[])

            :param agents: The list of agents to simulate.
            :type agents: List[Agent]

            :param envs: The list of systems which are not agents or disturbances to simulate.
            :type envs: List[System]

            :param duration: The length of simulation time to run the simulation for.
            :type duration: float

            :param dt: The interval of time to integrate systems over.
            :type dt: float

            :param obj_fun: An objective function for evaluating the performance of some aspect of the simulation. Not used in the current implementation - it has been temporarily removed from all methods, and will either be introduced later or completely removed.
            :type obj_fun: function

            :param disturbances: The list of disturbances to simulate.
            :type disturbances: List[DisturbanceSource]
        """
        self.agents = agents
        self.envs = envs
        self.disturbances = disturbances
        self.obj_fun = obj_fun

        self.t: float = 0
        self.ts: List[float] = [0]
        self.run_completed: bool = False

        self.duration = duration
        self.dt = dt


    def get_systems(self) -> List[System]:
        """
            Get the list of systems in the simulation. This is here for convenience, e.g. for an :class:`Animator` to get the list of systems which it should draw.
        """
        return self.envs + self.agents

    def reset(self) -> None:
        """
            A method to reset a :class:`Simulator`, so that it can be started again from the same initial conditions.
        """
        self.t: float = 0
        self.ts: List[float] = [0]
        self.run_completed: bool = False

        for agent in self.agents:
            agent.reset()
            agent.init_conditions()
        for env in self.envs:
            env.reset()
            env.init_conditions()
        for dist in self.disturbances:
            dist.reset()

    def perturb(self) -> None:
        for agent in self.agents:
            agent.perturb()
        for env in self.envs:
            env.perturb()

    def get_data(self):
        """
            A method for getting a :class:`Simulator`'s data. This will include timestamps, as well as the data of all simulated systems, except for that of the :class:`DisturbanceSource` s (this may be added in a later implementation).
        """
        agents_data = []
        for agent in self.agents:
            agents_data.append(agent.get_data())
        envs_data = []
        for env in self.envs:
            envs_data.append(env.get_data())

        return {"agents": agents_data, "envs": envs_data, "ts": self.ts}

    def step_forwards(self) -> None:
        """
            Step the simulatin forwards in time, by stepping all of its systems, using the simulations ``dt`` parameter.
        """
        # begin simulation main loop
        if self.t < self.duration:

            # step all robots
            for agent in self.agents:
                agent.step(self.dt)

            # step all environmental features
            for f in self.envs:
                f.step(self.dt)

            # step disturbance
            for disturbance in self.disturbances:
                disturbance.step(self.dt)

            # increment time variable and store in ts list for plotting later
            self.t += self.dt
            self.ts.append(self.t)
        else:
            self.run_completed = True

    def run(self) -> None:
        """
            A method for running the simulation once, by calling ``step_forwards`` repeatedly until the simulation's duration has been reached.
        """
        while self.t < self.duration:
            self.step_forwards()
