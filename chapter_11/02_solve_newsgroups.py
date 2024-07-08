from deap import base
from deap import creator
from deap import tools

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from newsgroup_classifier import NewsgroupClassifier

import elitism

# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the desired traveling salesman problem instace:
SUBSET_SIZE = 100  #10  # number of features
ngc = NewsgroupClassifier(RANDOM_SEED)

# Genetic Algorithm constants:
POPULATION_SIZE = 200
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.4   # probability for mutating an individual

def cxSubset(ind1, ind2): #TODO move
    """
    Executes a specialized crossover on the input individuals that represent a subset of n indices each,
    while ensuring that both offspring remain valid.
    For example: the parents are [5, 42, 88] and [73, 11, 42]
    1. Establish a set containing all unique indices found in both parent solutions: {5, 11, 42, 73, 88}
    2. Generate each offspring by randomly selecting n items from this set, e.g. [5, 11, 88] and [11, 42, 88]
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """
    mix = list(set(ind1 + ind2))
    ind1[:] = array.array("i", random.sample(mix, len(ind1)))
    ind2[:] = array.array("i", random.sample(mix, len(ind2)))
    return ind1, ind2

def mutSubset(individual, indpb):
    """
    Executes a specialized mutation on the input individual that represent a subset of n indices
    while ensuring that the resulting individual remains valid.
    For example: the individual is [5, 42, 88]
    for each item, with a probability indpb, the item will be replaced with one that does exist in the current list
    for example, the second item in the list above will be replaced with 11, resulting with [5, 11, 88]
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            while True:
                new_value = random.randrange(len(ngc))
                if new_value not in individual:
                    individual[i] = new_value
                    break

    return individual,


toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)

# create an operator that generates randomly shuffled indices:
toolbox.register("randomOrder", random.sample, range(len(ngc)), SUBSET_SIZE)

# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
def get_score(individual):
    return ngc.get_f1_score(individual),  # return a tuple


toolbox.register("evaluate", get_score)

# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", cxSubset)
toolbox.register("mutate", mutSubset, indpb=1.0/SUBSET_SIZE)

# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best individual info:
    best = hof.items[0]
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    print("-- Features subset selected = ")
    for i, j in enumerate(best):
        print(f"{i + 1}:    {j} = {ngc.get_feature_name(j)}")

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness vs. Generation')
    plt.show()


if __name__ == "__main__":
    main()
