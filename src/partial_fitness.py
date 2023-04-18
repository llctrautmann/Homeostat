import numpy as np
import random
from fitness_homeostat import *
import copy

def evaluate_fitness(pop_member, unit_instances):
    homeo = Homeostat(n_units=2, upper_viability=1, lower_viability=-1, upper_limit=10, lower_limit=-10, adapt_fun=None, adapt_enabled=False, test_interval=10, weights_set=None,adaptation_cooldown=400)

    homeo = copy.deepcopy(homeo)
    homeo.load_unit_instances(unit_instances=unit_instances)

    print(unit_instances[0]['thetas'][-1])

    homeo.adaptation_cooldown=500
    homeo.update_weights(weights=pop_member)



    t = 0
    ts = [t]
    dt = 0.01
    duration = 50

    maes = []

    while t < duration: # wahala
        homeo.step(dt)
        t += dt
        ts.append(t)
        theta = homeo.get_theta()

        mae = [abs(theta - 0) for theta in theta]
        maes.append(mae)

    fitness = np.sum(maes) / len(maes)

    # print(f'the fitness for this solution is {mse}')

    return fitness





