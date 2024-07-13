from deap import base
from deap import creator
from deap import tools

import random
import numpy
import os

import image_test
import elitism_and_callback

import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing

# problem related constants
POLYGON_SIZE = 3
NUM_OF_POLYGONS = 15 # 20

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 5000
HALL_OF_FAME_SIZE = 10
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
MONA_LISA_PATH = os.path.join(BASE_PATH, "Mona_Lisa_head.png")
RESULTS_PATH = os.path.join(BASE_PATH, "results", "blur", f"run-{POLYGON_SIZE}-{NUM_OF_POLYGONS}")

# create the image test class instance:
imageTest = image_test.ImageTest(MONA_LISA_PATH, POLYGON_SIZE)

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = NUM_OF_POLYGONS * (POLYGON_SIZE * 2 + 4)

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

# create an operator that randomly returns a float in the desired range:
toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

# create an operator that fills up an Individual instance:
toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

# create an operator that generates a list of individuals:
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)


# fitness calculation using the current difference metric:
def getDiff(individual):
    return imageTest.getDifference(individual, blur=True),

toolbox.register("evaluate", getDiff)


# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0/NUM_OF_PARAMS)


# save the best current drawing every 100 generations (used as a callback):
def saveImage(gen, polygonData):

    # only every 100 generations:
    if gen % 100 == 0:

        # create folder if does not exist:
        if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # save the image in the folder:
        imageTest.saveImage(polygonData, 
                            os.path.join(RESULTS_PATH, f"after-{gen}-gen.png"),
                            f"After {gen} Generations")

# Genetic Algorithm flow:
def main():

    with multiprocessing.Pool(processes=20) as pool:
        toolbox.register("map", pool.map)

        # create initial population (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min)
        stats.register("avg", numpy.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)


        # perform the Genetic Algorithm flow with elitism and 'saveImage' callback:
        population, logbook = elitism_and_callback.eaSimpleWithElitismAndCallback(population,
                                                        toolbox,
                                                        cxpb=P_CROSSOVER,
                                                        mutpb=P_MUTATION,
                                                        ngen=MAX_GENERATIONS,
                                                        callback=saveImage,
                                                        stats=stats,
                                                        halloffame=hof,
                                                        verbose=True)

    # print best solution found:
    best = hof.items[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()

    # draw best image next to reference image:
    imageTest.plotImages(imageTest.polygonDataToImage(best))

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.figure("Stats:")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()

if __name__ == "__main__":
    main()
