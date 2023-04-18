import sys
# relative path to folder which contains Sandbox
sys.path.insert(1, '..')
from Sandbox import *
import unittest
from testcase import MyTestCase

'''

    The Controller class has these attributes:

        noisemakers
        noisemakers_inds
        step_fun
        adapt_fun

        inputs_hist
        initial_inputs_hist
        commands_hist
        initial_commands_hist

        params
        params_hist
        initial_params

    The Controller class has these attributes:

        __init__
        step
        reset
        get_data


'''

def dummy_stepfun(dt, inputs, params, state):

    return [0, 1, 2, 3, 4]

class Test_Controller(MyTestCase):

    def test_init(self) -> None:

        print("Testing Controller")

        c = Controller(inputs_n = 5, commands_n = 3, step_fun = dummy_stepfun)

        self.assertTrue(c.noisemakers == None)
        self.assertTrue(c.noisemakers_inds == None)
        self.assertTrue(c.params == None)
        self.assertTrue(c.params_hist == None)
        self.assertTrue(c.initial_params == None)
        self.assertTrue(c.inputs_hist == [[0,0,0,0,0]])
        self.assertTrue(c.initial_inputs_hist == c.inputs_hist)
        self.assertTrue(c.commands_hist == [[0,0,0]])
        self.assertTrue(c.initial_commands_hist == c.commands_hist)


        c = Controller(inputs_n=3, commands_n=5, step_fun=dummy_stepfun, params=[9,6,3])

        self.assertTrue(c.noisemakers == None)
        self.assertTrue(c.noisemakers_inds == None)
        self.assertTrue(c.params == [9,6,3])
        self.assertTrue(c.params_hist == [[9,6,3]])
        self.assertTrue(c.initial_params == c.params)
        self.assertTrue(c.inputs_hist == [[0,0,0]])
        self.assertTrue(c.initial_inputs_hist == c.inputs_hist)
        self.assertTrue(c.commands_hist == [[0,0,0,0,0]])
        self.assertTrue(c.initial_commands_hist == c.commands_hist)

    def test_step(self):

        c = Controller(inputs_n=3, commands_n=5, step_fun=dummy_stepfun, params=[9,6,3])

        n = 10
        inp = [2.1, 3.5, 4.7]
        for _ in range(n):

            c.step(0.1, inputs=inp, state=[7,4])

        self.assertTrue(len(c.inputs_hist) == n+1)
        self.assertTrue(len(c.commands_hist) == n+1)
        self.assertTrue(len(c.params_hist) == n+1)

    def test_get_data_and_reset(self):

        c = Controller(inputs_n=3, commands_n=5, step_fun=dummy_stepfun, params=[9,6,3])

        n = 10
        inp = [2.1, 3.5, 4.7]
        for _ in range(n):

            c.step(0.1, inputs=inp, state=[7,4])

        self.assertTrue(len(c.inputs_hist) == n+1)
        self.assertTrue(len(c.commands_hist) == n+1)
        self.assertTrue(len(c.params_hist) == n+1)

        d = c.get_data_and_reset()

        self.assertTrue(c.noisemakers == None)
        self.assertTrue(c.noisemakers_inds == None)
        self.assertTrue(c.params == [9,6,3])
        self.assertTrue(c.params_hist == [[9,6,3]])
        self.assertTrue(c.initial_params == c.params)
        self.assertTrue(c.inputs_hist == [[0,0,0]])
        self.assertTrue(c.initial_inputs_hist == c.inputs_hist)
        self.assertTrue(c.commands_hist == [[0,0,0,0,0]])
        self.assertTrue(c.initial_commands_hist == c.commands_hist)

        self.assertTrue(len(d["params_hist"]) == n+1)
        self.assertTrue(len(d["inputs_hist"]) == n+1)
        self.assertTrue(len(d["commands_hist"]) == n+1)

        self.assertTrue(len(d["params_hist"][-1]) == 3)
        self.assertTrue(len(d["inputs_hist"][-1]) == 3)
        self.assertTrue(len(d["commands_hist"][-1]) == 5)
