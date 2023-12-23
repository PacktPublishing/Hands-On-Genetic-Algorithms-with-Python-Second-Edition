from deap import base
from deap import creator
from deap import tools

import elitism

import random
import numpy
import time

# problem constants:
ONE_MAX_LENGTH = 10  # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
MAX_GENERATIONS = 5
HALL_OF_FAME_SIZE = 3

DELAY_SECONDS = 3

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def busy_wait(duration):
    current_time = time.time()
    while (time.time() < current_time + duration):
        pass
               
# fitness calculation:
# compute the number of '1's in the individual
def oneMaxFitness(individual):
    busy_wait(DELAY_SECONDS)
    return sum(individual),  # return a tuple

toolbox.register("evaluate", oneMaxFitness)

# Tournament selection with tournament size of 4:
toolbox.register("select", tools.selTournament, tournsize=4)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, _ = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print("Best Individual = ", hof.items[0])

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time = {(end - start):.2f} seconds")
