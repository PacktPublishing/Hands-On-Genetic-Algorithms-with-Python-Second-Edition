import random
import numpy
import numpy as np

from deap import base
from deap import creator
from deap import tools

from mystery_word_game import MysteryWordGame
from embeddings import Embeddings
from elitism_modified import eaSimple_modified

# Genetic Algorithm constants:
POPULATION_SIZE = 30 
MAX_GENERATIONS = 1000
HALL_OF_FAME_SIZE = 3
P_CROSSOVER = 0.9       # probability for crossover
P_MUTATION = 0.8        # probability for mutating an individual
CROWDING_FACTOR = 0.01  # crowding factor for crossover and mutation
MAX_SCORE = 100         # max possible score in the game

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the embedding model:
embeddings = Embeddings(model_name='glove-wiki-gigaword-50', randomSeed=RANDOM_SEED)
VECTOR_SIZE = embeddings.get_vector_size()

# create a game instance:
game = MysteryWordGame(given_mystery_word='dog')

# boundaries for vector entries:
BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0  # boundaries for all dimensions

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * VECTOR_SIZE, [up] * VECTOR_SIZE)]

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


# fitness calculation using similarity score:
def score(individual):
    guess_word = embeddings.vec2_nearest_word(np.asarray(individual))
    return game.score_guess(guess_word),


toolbox.register("evaluate", score)


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
                 indpb=1.0 / VECTOR_SIZE)


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
    population, logbook = eaSimple_modified(embeddings,
                                            population,
                                            toolbox,
                                            cxpb=P_CROSSOVER,
                                            mutpb=P_MUTATION,
                                            ngen=MAX_GENERATIONS,
                                            max_fitness=MAX_SCORE,
                                            stats=stats,
                                            halloffame=hof,
                                            verbose=True)

    # print best solution found:
    best = hof.items[0]
    print()
    print(f"Best Solution = {embeddings.vec2_nearest_word(np.asarray(best))}")
    print(f"Best Score = {best.fitness.values[0]:.2f}")
    print()


if __name__ == "__main__":
    main()