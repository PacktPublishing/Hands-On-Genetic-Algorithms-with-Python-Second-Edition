from deap import tools
from deap import algorithms

import numpy as np

def eaSimple_modified(embeddings, population, toolbox, cxpb, mutpb, ngen, 
                        max_fitness = None, stats=None, halloffame=None, verbose=__debug__):
    """
    This algorithm is based on the DEAP eaSimple() algorithm, with the following modifications:
    - The halloffame is used to implement an elitism mechanism
    - The best individual in each generation is converted to a word and printed out
    - The main loop will break early if the value max_fitness is reached
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(f"{logbook.stream} => {embeddings.vec2_nearest_word(np.asarray(halloffame.items[0]))}")

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(f"{logbook.stream} => {embeddings.vec2_nearest_word(np.asarray(halloffame.items[0]))}")

        if max_fitness and halloffame.items[0].fitness.values[0] >= max_fitness:
            break

    return population, logbook

