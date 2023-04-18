import numpy as np
import random
from fitness_homeostat import *
import copy
import functools

def evaluate_fitness(pop_member, unit_instances):
    @functools.lru_cache(maxsize=None)
    def create_deep_copy(unit_instances):
        homeo = copy.deepcopy(unit_instances)
        return homeo

    
    homeo = create_deep_copy(unit_instances)


    homeo.update_weights(weights=pop_member)
    homeo.adaptation_cooldown=500



    t = 0
    ts = [t]
    dt = 0.01
    duration = 150

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





