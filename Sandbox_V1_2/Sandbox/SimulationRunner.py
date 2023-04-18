from .base import *
from .Simulator import *
from .Animator import *

# how to use:
#   create agents, envs, Simulator
#   create this
#   run this

class SimulationRunner:
    """
        A class for running a batch of simulations, which may or may not be animated, as required.
    """
    def __init__(self, sim: Simulator, animate: bool=False, screen_width: int=700, pause_ani=False, animation_delay: int=0, perturb: bool=True):
        """
            __init__(sim: Simulator, animate: bool=False, screen_width: int=700, pause_ani=False, animation_delay: int=0)

            :param sim: The :class:`Simulator` which will be used (note: future implementations will not need this to be passed in, as it can easily be constructed here).
            :type sim: Simulator

            :param animate: A flag which determines whether or not the simulation runs will be animated. Note: if this is set to ``False``, then no :class:`Animator` will be constructed, so it cannot be changed during simulation runs. When this flag is set to ``True``, the :class:`Animator` is constructed, and you can the choose to use it, or skip it, using its controls.
            :type animate: bool

            :param screen_width: The PyGame display height and width, in pixels.
            :type screen_width: int

            :param pause_ani: If this flag is set to ``True`` (which is it by default), and simulation runs are being animated, then animations will always begin in the paused state.
            :type pause_ani: bool

            :param delay: The number of 10s of milliseconds to delay the animation by, when one is used, inbetween each pair of successive simulation steps. Can be used to slow an animation/simulation down in order to see what is happening more clearly.
            :type delay: int
        """
        self.sim = sim
        self.animate = animate
        self.ani = None
        self.pause_ani = pause_ani
        if self.animate:
            self.ani = Animator(systems=sim.get_systems(), screen_width=screen_width, paused=pause_ani, delay=animation_delay)
        self.perturb = perturb

    def run_sims(self, n: int=1):
        """
            :param n: The number of simulation runs to execute.
            :type n: int

            :return: The data from all simulation runs, in a list of data dicts, as returned from :meth:`Sandbox.Simulator.get_data`
        """
        data = []

        # run set of simulations
        for i in range(n):
            self.sim.reset()
            if self.ani:
                self.ani.animate_current = True

            if self.perturb:
                self.sim.perturb()

            run_data = self.run_once()
            run_data["run_n"] = i

            data.append(run_data)

        # shutdown animator
        if self.ani:
            self.ani.shutdown()

        return data

    def run_once(self):
        """
            A method to run, and animate, a single simulation run.

            :return: Data from the simulation run, as returned from :meth:`Sandbox.Simulator.get_data`
            :rtype: dict
        """
        if self.ani:
            self.ani.paused = self.pause_ani

        while not self.sim.run_completed:
            # step simulation once
            # if simulation returns quit, tell animator to close window, and then return
            if self.sim.step_forwards():
                if self.ani:
                    self.ani.stop()
                return

            # step animation once, and then
            # continue to step if it is paused
            # if animation returns quit, then return

            # if Animator exists
            if self.ani:
                # begin with animate=True
                # - if the animation gets paused, then we continue to redraw,
                #   as that is the only way keyboard commands will be captured
                animate = True
                while animate and self.ani.animate_any and self.ani.animate_current and not self.sim.run_completed:
                    self.sim.run_completed = self.ani.draw_frame(self.sim.t)
                    # if the anmiation is not paused, then animate=False and we
                    # will exit this while loop
                    animate = self.ani.paused

        return self.sim.get_data()
