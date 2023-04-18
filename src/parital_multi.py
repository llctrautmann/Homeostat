import numpy as np
from fitness import *
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.3

def mutation(gene,mutation_rate=None):
    for i in range(len(gene)):
        if random.random() < mutation_rate:
            r = random.random()
            if r < 0.25:
                gene[i] = random.random()
            elif r < 0.5:
                gene[i] = random.random() * -1
            elif r < 0.75:
                gene[i] += random.random() * -3
            else:
                gene[i] -= random.random() * 3

        
    return gene 

def cx_point(loser,winner,pcx=CROSSOVER_RATE):
    loser = loser
    winner = winner

    for i in range(len(winner)):
        if random.random() <  pcx:
            loser[i] = winner[i]      

    return loser

# def solid_state_evolution(pop,fitness,unit_instance, cx_function=cx_point):
#     # selection of random genes
#     selector = np.random.randint(0,len(pop)-1)
#     selector_2 = (selector + np.random.randint(1,7)) % len(pop)

#     # Index Fitness Vector

#     fitness_1 = evaluate_fitness(pop_member=pop[selector],unit_instances=unit_instance)
#     fitness_2 = evaluate_fitness(pop_member=pop[selector_2],unit_instances=unit_instance)

#     fitness[selector] = fitness_1 
#     fitness[selector_2] = fitness_2

#     # mutation, crossover

#     if fitness_1 >= fitness_2:
#         pop[selector] = cx_function(loser=pop[selector],winner=pop[selector_2])
#         pop[selector] = mutation(pop[selector])
#     else:
#         pop[selector_2] = cx_function(loser=pop[selector_2],winner=pop[selector])
#         pop[selector_2] = mutation(pop[selector_2])


def solid_state_evolution(pop,fitness,unit_instance, cx_function=cx_point,mutation_rate=None,crossover_rate=None):
    # selection of random genes
    selector = np.random.randint(0,len(pop)-1)
    selector_2 = (selector + np.random.randint(1,7)) % len(pop)

    # Index Fitness Vector

    fitness_1 = evaluate_fitness(pop_member=pop[selector],unit_instances=unit_instance)
    fitness_2 = evaluate_fitness(pop_member=pop[selector_2],unit_instances=unit_instance)

    fitness[selector] = fitness_1 
    fitness[selector_2] = fitness_2

    # mutation, crossover

    if fitness_1 >= fitness_2:
        pop[selector] = cx_function(loser=pop[selector],winner=pop[selector_2],pcx=crossover_rate)
        pop[selector] = mutation(pop[selector],mutation_rate=mutation_rate)
    else:
        pop[selector_2] = cx_function(loser=pop[selector_2],winner=pop[selector],pcx=crossover_rate)
        pop[selector_2] = mutation(pop[selector_2],mutation_rate=mutation_rate)



def linear_ga(n_genes, homeostat_cp,mutation_rate,crossover_rate):
    # 1. initialize population
    pop =  np.random.uniform(low=-2, high=2, size=(15,n_genes))
    fitness = np.full(15, 1000)
    generations = 50

    min_fitness = 1000
    weights = None

    

    for gen in range(generations): # maybe this is very unecessary and I can only spin up new machines for the two genes in question 
        if min_fitness < 1:
            break
        # print(f' gen = {gen} - fitness = {fitness}')
        solid_state_evolution(pop=pop,fitness=fitness,unit_instance=homeostat_cp,mutation_rate=mutation_rate,crossover_rate=crossover_rate)

        if np.min(fitness) < min_fitness:
            min_fitness = np.min(fitness)
            weights = pop[np.argmin(fitness)]

        else:
            pass
    

    return weights.tolist()


def linear_pso(n_genes, homeostat_cp, n_particles=20, n_iterations=50, w=0.5, c1=1, c2=2):
    # 1. Initialize particles and velocities
    particles = np.random.uniform(low=-2, high=2, size=(n_particles, n_genes))
    velocities = np.random.uniform(low=-1, high=1, size=(n_particles, n_genes))

    # 2. Evaluate initial fitness
    fitness_values = [evaluate_fitness(pop_member, unit_instances=homeostat_cp) for pop_member in particles]

    # 3. Initialize personal and global bests
    p_bests = particles.copy()
    p_bests_fitness = fitness_values.copy()
    g_best = particles[np.argmin(p_bests_fitness)]
    g_best_fitness = np.min(p_bests_fitness)

    # 4. Main loop

    for iteration in range(n_iterations):

        print(f'gen {iteration}')
        if min(fitness_values) < 1:
            break
        for i in range(n_particles):
            # Update particle velocity
            r1, r2 = np.random.rand(2, n_genes)
            velocities[i] = w * velocities[i] + c1 * r1 * (p_bests[i] - particles[i]) + c2 * r2 * (g_best - particles[i])

            # Update particle position
            particles[i] += velocities[i]

        # Evaluate fitness
        fitness_values = [evaluate_fitness(pop_member, unit_instances=homeostat_cp) for pop_member in particles]

        # Update personal and global bests
        for i in range(n_particles):
            if fitness_values[i] < p_bests_fitness[i]:
                p_bests[i] = particles[i].copy()
                p_bests_fitness[i] = fitness_values[i]

                if p_bests_fitness[i] < g_best_fitness:
                    g_best = p_bests[i].copy()
                    g_best_fitness = p_bests_fitness[i]

        w -= 0.008

    print(f'best fitness: {g_best_fitness}')

    return g_best.tolist()






if __name__ == '__main__':
    linear_ga()


