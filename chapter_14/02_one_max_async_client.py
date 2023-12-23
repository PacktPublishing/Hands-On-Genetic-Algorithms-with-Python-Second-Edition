from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import elitism_async

import random
import numpy
import time

import asyncio

# problem constants:
ONE_MAX_LENGTH = 10  # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
MAX_GENERATIONS = 5
HALL_OF_FAME_SIZE = 3

BASE_URL="http://127.0.0.1:5000/"

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

# fitness calculation done by server:
# compute the number of '1's in the individual
async def async_oneMaxFitness_client(session, individual):
    individual_as_str = ''.join(str(bit) for bit in individual)
    url = f'{BASE_URL}/one_max_fitness/{individual_as_str}'
    async with session.get(url) as resp:
        sum_digits_str = await resp.text()
        return int(sum_digits_str),  # return a tuple
    
toolbox.register("evaluate", async_oneMaxFitness_client)

# Tournament selection with tournament size of 4:
toolbox.register("select", tools.selTournament, tournsize=4)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


# Genetic Algorithm flow:
async def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, _ = await elitism_async.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                            ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print("Best Individual = ", hof.items[0])

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Elapsed time = {(end - start):.2f} seconds")