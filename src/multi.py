import numpy as np
from fitness import *
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.3

def mutation(gene):
    for i in range(len(gene)):
        r = random.random()
        if r < 0.25:
            gene[i] = random.random()
        elif r < 0.5:
            gene[i] = random.random() * -1
        elif r < 0.75:
            gene[i] += random.random() * -2 
        else:
            gene[i] -= random.random() + 2

        
    return gene 

def cx_point(loser,winner,pcx=CROSSOVER_RATE):
    loser = loser
    winner = winner

    for i in range(len(winner)):
        if random.random() <  pcx:
            loser[i] = winner[i]      

    return loser

def solid_state_evolution(pop,fitness, cx_function=cx_point):
    # selection of random genes
    selector = np.random.randint(0,len(pop)-1)
    selector_2 = (selector + np.random.randint(1,7)) % len(pop)

    # Index Fitness Vector

    fitness_1 = fitness[selector]
    fitness_2 = fitness[selector_2]

    # mutation, crossover

    if fitness_1 >= fitness_2:
        pop[selector] = cx_function(loser=pop[selector],winner=pop[selector_2])
        pop[selector] = mutation(pop[selector])
    else:
        pop[selector_2] = cx_function(loser=pop[selector_2],winner=pop[selector])
        pop[selector_2] = mutation(pop[selector_2])


def ga(n_genes, unit_instances):
    print(f'{n_genes} weights need to be updated.')
    # 1. initialize population
    pop =  np.random.uniform(low=-2, high=2, size=(25,n_genes))
    fitness = np.zeros(shape=(25, 1))
    generations = 50

    min_fitness = 1000
    weights = None

    for gen in range(generations): # maybe this is very unecessary and I can only spin up new machines for the two genes in question 
        # Partial Function to feed into the fitness function
        partial_evaluate_fitness = partial(
            evaluate_fitness,
            unit_instances=unit_instances
        )

        with Pool() as pool:
            fitness = pool.map(partial_evaluate_fitness, pop)

            if np.min(fitness) < min_fitness:
                min_fitness = np.min(fitness)
                weights = pop[np.argmin(fitness)]

            else:
                pass

    
        # Rest of the GA process (selection, crossover, mutation, etc.) should be implemented here
        solid_state_evolution(pop=pop, fitness=fitness)
    print(f'the minimum cost is {min_fitness}')

    

    return weights.tolist()

if __name__ == '__main__':
    ga()


