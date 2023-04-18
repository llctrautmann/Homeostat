import sys
import random
import numpy as np
# relative path to folder which contains the Sandbox module
sys.path.insert(1, './Sandbox_v1_2')
from Sandbox import *


'''
    A class to simulate a single unit in a simulation of Ashby's Homeostat machine.
'''
class Unit(System):

    def __init__(self, test_interval, adapt_fun, upper_viability=1, lower_viability=-1, upper_limit=np.Inf, lower_limit=-np.Inf, m=1, k=1, l=1, p=2, q=1, theta0=2, theta_dot0=0, weights_set=None, adapt_enabled=True):

        # Ashby's homeostat used a discrete set of weights. If we want
        # to do that, we need to pass a list of weights into here, and
        # our adapt_fun needs to be written to work with them (e.g. see
        # random_selector)
        self.weights_set = weights_set
        # minimum time to wait between changing the weights
        self.test_interval = test_interval
        # clock
        self.t = 0
        # system equation parameters
        self.m = m
        self.k = k
        self.l = l
        self.p = p
        self.q = q
        # "physical" upper limit for theta - the variable (or needle) can't move past this point
        self.upper_limit = upper_limit
        # "physical" lower limit for theta - the variable (or needle) can't move past this point
        self.lower_limit = lower_limit
        # upper limit for viability
        self.upper_viability = upper_viability
        # lower limit for viability
        self.lower_viability = lower_viability

        self.thetas = [theta0] # the system variable. theta is used instead of x, because in SituSim x is always used to mean a spatial coordinate
        self.theta_dots = [theta_dot0] # the system state is [theta, theta_dot]
        self.theta_dotdots = [0] # the input to the system is converted into an acceleration (technically, it is a force, but mass is not modelled here)
        self.units = [] # a list of connected Units
        self.weights = [] # a list of weights, which are applied to connected units
        self.weights_hist = [] # a record of weights over time
        self.testing = False # this boolean variable keeps track of whether the system is in the process of adaptation
        self.testing_hist = [self.testing] # a record of when the unit is adapting
        self.inputs_hist = [] # a record of all inputs to a unit over time, including the feedback from the unit itself
        self.adapt_fun = adapt_fun #
        self.self_ind = 0 # used to be able to set the unit's feedback connection to always be negative
        self.test_times = [] # times when the unit starts testing new weights
        self.timer = 0 # a timer for how long to wait after changing a unit's weights
        self.adapt_enabled = adapt_enabled # Unit will only adapt if this variable is set to True


    def get_instance_variables(self):
        return {key: value for key, value in self.__dict__.items()}

    def update_instance_variables(self, variables_dict):
        for key, value in variables_dict.items():
            if key in self.__dict__:
                setattr(self, key, value)



    # this method should be called after a unit's connections have
    # been made, and before it is run
    def initialise(self):
        self.weights_hist.append(self.weights)

    # randomise the parameters which affect the unit's dynamics
    # (i.e. all parameters *apart from* the connection weights)
    def randomise_params(self):

        self.m = random_in_interval(minimum=0.1, maximum=2)
        self.k = random_in_interval(minimum=0.1, maximum=2)
        self.l = random_in_interval(minimum=0.1, maximum=2)
        self.q = random_in_interval(minimum=0.1, maximum=1)
        self.p = self.q + random_in_interval(minimum=0.1, maximum=2)

        # print("***** Randomising homeostat unit params *****")
        # print("m", self.m)
        # print("k", self.k)
        # print("l", self.l)
        # print("p", self.p)
        # print("q", self.q)

    # step unit forwards in time
    def step(self, dt):
        # get weighted sum of inputs
        input_sum = self.update_inputs(dt)
        # integrate the system's dynamics
        self.integrate(dt, input_sum)

        # manage timer
        if self.timer > self.test_interval:
            # reset timer
            self.timer = 0
            self.testing = False

        # if not viable
        # if self.adapt_enabled and not self.test_viability():
        #     # not viable
        #     # if already testing new parameters, then continue to test
        #     # otherwise, try some new weights
        #     if not self.testing:
        #         self.adjust_weights(dt)
        #         # start testing
        #         self.testing = True
        #         # keep record of times when new weights are set
        #         self.test_times.append(self.t)
        #         # reset timer
        #         self.timer = 0

        # keep record of when the system is testing new weights
        self.testing_hist.append(self.testing)
        # keep history of weights over time
        self.weights_hist.append(self.weights)

        # increment clock
        self.t += dt
        # increment test timer
        self.timer += dt

        # return current state
        return self.thetas[-1]

    # get inputs from all connect units, including this unit
    def update_inputs(self, dt):

        # calculate weighted sum of inputs from all connected Units (including feedback from this Unit)
        input_sum = 0
        inputs = []
        for unit, weight in zip(self.units, self.weights):
            inputs.append(unit.get_theta() * weight)
            input_sum += inputs[-1]
        # keep record of inputs
        self.inputs_hist.append(inputs)

        return input_sum

    # integrate system's dynamics
    def integrate(self, dt, input_sum):

        # integrate the system, from acceleration to position
        theta_dotdot = ((-self.k * self.theta_dots[-1]) + (self.l * (self.p - self.q) * input_sum)) / self.m # calculate acceleration
        # - we integrate twice here, because this is a second order system
        theta_dot = (self.theta_dots[-1] + (self.theta_dotdots[-1] * dt)) # integrate acceleration to get velocity
        theta = (self.thetas[-1] + (self.theta_dots[-1] * dt)) # integrate velocity to get position

        # in Ashby's Homeostat, there were hard limits to how far the needle
        # (system variable) could move in either direction - enforce these limits
        #   If you don't want to enforce limits, you can leave them at them
        # at the default values of +-np.Inf
        if theta > self.upper_limit:
            theta = self.upper_limit
            theta_dot = 0
        elif theta < self.lower_limit:
            theta = self.lower_limit
            theta_dot = 0

        # store system variable and its first and second derivatives
        self.thetas.append(theta)
        self.theta_dots.append(theta_dot)
        self.theta_dotdots.append(theta_dotdot)

    # test to see whether the system's essential variable (theta) is
    # within the chosen limits for viability
    def test_viability(self):

        # test whether  the Unit is within viable limits
        if ((self.thetas[-1] > self.upper_viability) or
            (self.thetas[-1] < self.lower_viability)):
            return False # return False for not viable
        return True # return True for viable

    # adjust weights
    def adjust_weights(self, dt):

        # adjust parameters
        self.weights = self.adapt_fun(dt, self.inputs_hist, self.weights_hist, self.thetas, self.theta_dots, weights_set=self.weights_set, self_ind=self.self_ind)

    # connect a Unit to this Unit
    def add_connection(self, unit, weight):
        # add Unit to list
        self.units.append(unit)
        # add connection weight to weights list
        self.weights.append(weight)

        # if adding a connection from the unit to itself, keep
        # track of the index of the weight
        if unit == self:
            self.self_ind = len(self.units) - 1

    # get the state of the Unit variable (the full state of a Unit is
    # actually [theta, theta_dot], but other Units can only "see" theta)
    def get_theta(self):
        return self.thetas[-1]
    
    def get_weights(self):
        return self.weights
    
    def get_thetas(self):
        return np.mean(self.thetas)


'''
    A class to simulate Ashby's Homeostat machine.
'''
class Homeostat(System):

    def __init__(self, n_units, upper_viability, lower_viability, adapt_fun, upper_limit=np.Inf, lower_limit=-np.Inf, weights_set=None, test_interval=10, adapt_enabled=True,adaptation_cooldown=50):

        # set up units
        self.units = []
        for _ in range(n_units):
            self.units.append(Unit(test_interval=test_interval, adapt_fun=adapt_fun, upper_limit=upper_limit, lower_limit=lower_limit, upper_viability=upper_viability, lower_viability=lower_viability, weights_set=weights_set, adapt_enabled=adapt_enabled))

        # connect units, with random weights
        for unit in self.units:
            for unit2 in self.units:
                unit.add_connection(unit2, random_in_interval(minimum=-1, maximum=1))

        # initialise units
        self.initialise()
        # initialise Homeostat's time variable
        self.t = 0
        # Adaptation cooldown time
        self.adaptation_cooldown = adaptation_cooldown
        self.time_since_adaptation = 0

    # step unit forwards in time
    # def step(self, dt):
    #     for unit in self.units:
    #         unit.step(dt)

    #     self.t += dt


    def step(self, dt):
       
        not_viable_units = []
        for unit in self.units:
            unit.step(dt)
            if not unit.test_viability():
                not_viable_units.append(unit)

        if not_viable_units and self.units[0].adapt_enabled and self.time_since_adaptation >= self.adaptation_cooldown:
            self.adjust_weights(dt, not_viable_units)
            self.time_since_adaptation = 0
        else:
            self.time_since_adaptation += dt

        self.t += dt

    def adjust_weights(self, dt, not_viable_units):
        # all_new_weights = self.units[0].adapt_fun(dt, self.units[0].inputs_hist, self.units[0].weights_hist, self.units[0].thetas, self.units[0].theta_dots, weights_set=self.units[0].weights_set, self_ind=self.units[0].self_ind)
        all_new_weights = [random.uniform(-1, 1) for _ in range(len(not_viable_units) * 4)]
        # all_new_weights = ga(n_genes=len(not_viable_units))

        # Calculate the number of weights per unit
        weights_per_unit = len(all_new_weights) // len(not_viable_units)

    
        # Break down the new_weights list into pieces for each not viable unit
        for i, unit in enumerate(not_viable_units):
            unit_new_weights = all_new_weights[i * weights_per_unit:(i + 1) * weights_per_unit]
            unit.weights = unit_new_weights


    def all_weights(self):
        weights = []
        for idx, unit in enumerate(self.units):
            weights.append(f' weights unit {idx}:  {unit.get_weights()}')
        # print(f'the current unit weights are {weights}')

    def get_theta(self):
        thetas = []
        for idx, unit in enumerate(self.units):
            thetas.append(unit.get_theta())
        
        return thetas


    def get_all_thetas(self):
        all_thetas = []
        for idx, unit in enumerate(self.units):
            all_thetas.append(unit.get_thetas())
        # print(all_thetas)

        return all_thetas


    # initialise units
    def initialise(self):
        for unit in self.units:
            unit.initialise()

    # randomise unit parameters (this does not modify connection weights)
    def randomise_params(self):
        for unit in self.units:
            unit.randomise_params()

    def get_unit_instances(self) -> list:
        unit_instances = []
        for unit in self.units:
            unit_instances.append(unit.get_instance_variables())

        return unit_instances
    
    def load_unit_instances(self, unit_instances):
        for idx, unit in enumerate(self.units):
            unit.update_instance_variables(unit_instances[idx])

    def adjust_cooldown(self, cooldown):
        self.adaptation_cooldown = cooldown

    def update_weights(self, weights):
        disturbed_units = 0
        not_viable_units = []

        for unit in self.units:
            if unit.get_theta() < -1 or unit.get_theta() > 1:
                disturbed_units += 1
                not_viable_units.append(unit)

        # print(f'number of units under change = {disturbed_units}')

        weights_per_unit = len(weights) // disturbed_units

        for i, unit in enumerate(not_viable_units):
            unit_new_weights = weights[i * weights_per_unit:(i + 1) * weights_per_unit]
            # print(f' the new unit weights are: {unit_new_weights}')
            unit.weights = unit_new_weights


'''
	An example adaptation function, loosely based on Ashby's random step change. It differs from Ahsby's mechanism in that it does
	not choose weights from a discrete set of values, but randomly
	draws values from a uniform interval.
'''
def random_val(dt, inputs_hist, weights_hist, thetas, theta_dots, weights_set=[], self_ind = None):
    weights = []
    for _ in range(len(weights_hist[0])):
        weights.append(random_in_interval(-1, 1))

    # this ensures that the self-connection on a Unit will have a
	# negative weight - this is not a requirement, but will lead to
	# stability being found quicker
    if self_ind is not None:
        weights[self_ind] = - np.abs(weights[self_ind])

    # return new weights
    return weights

'''
	An example adaptation function, which moves weights by a small
	random amount from their current values.
'''
def random_creeper(dt, inputs_hist, weights_hist, thetas, theta_dots, weights_set=[], self_ind = None):

    weights = []
    for i in range(len(weights_hist[-1])):
        weights.append(weights_hist[-1][i] + random_in_interval(-0.05, 0.05))

    # this ensures that the self-connection on a Unit will have a
	# negative weight - this is not a requirement, but will lead to
	# stability being found quicker
    if self_ind:
        weights[self_ind] = - np.abs(weights[self_ind])

    # return new weights
    return weights

'''
	An example adaptation function, loosely based on Ashby's random step change. Like Ashby's mechanism it chooses weights from a discrete set of values, "weights_set".
'''
def random_selector(dt, inputs_hist, weights_hist, thetas, theta_dots, weights_set, self_ind = None):

    weights = []
    for _ in range(len(weights_hist[0])):
        weights.append(np.random.choice(weights_set))

    # this ensures that the self-connection on a Unit will have a
	# negative weight - this is not a requirement, but will lead to
	# stability being found quicker
    if self_ind:
        weights[self_ind] = - np.abs(weights[self_ind])

    # return new weights
    return weights
