from deap import base
from deap import creator
from deap import tools

import random
import numpy

from credit_risk_data import CreditRiskData
import elitism

# Genetic Algorithm constants:
POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2   # probability for mutating an individual
MAX_GENERATIONS = 30
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

PENALTY = 100 
NUM_OF_PARAMS = 4

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create an instance of the Crefdit test class:
credit_data = CreditRiskData(randomSeed=RANDOM_SEED)

# placeholders:
bounds_low =  []
bounds_high = []
ranges = []

def set_ranges(amount_low, amount_high, duration_low, duration_high, checking_low, checking_high, savings_low, savings_high): 
  
    bounds_low =  [amount_low, duration_low, checking_low, savings_low] 
    bounds_high = [amount_high, duration_high, checking_high, savings_high]
    
    bounds_high = [high + 1 for high in bounds_high]
    ranges = [high - low for high, low in zip(bounds_high, bounds_low)]

    return bounds_low, bounds_high, ranges

# boundaries for attributes:
# "amount":     100..5000
# "duration":   2..72
# "checking":   0..3 
# "savings":    0..4
bounds_low, bounds_high, ranges = set_ranges(100, 5000, 2, 72, 0, 3, 0, 4)

# take a applicant from one of the rows of the dataset:
applicant = credit_data.get_applicant(25) 

# extract the original values of the four attributes:
applicant_values = applicant.get_values()

# calculate the cost of the new attribute values based on their differences from the original ones:
def get_cost(individual):
    cost = sum([abs(int(individual[i]) - applicant_values[i])/ranges[i] for i in range(NUM_OF_PARAMS)])

    # penalize if prediction indicates risk:
    if credit_data.is_credit_risk(applicant.with_values(individual)):
        cost += PENALTY * credit_data.risk_probability(applicant.with_values(individual))

    return cost,

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# create an operator that returns a random value between the bounds:
toolbox.register("amount", random.uniform, bounds_low[0], bounds_high[0])
toolbox.register("duration", random.uniform, bounds_low[1], bounds_high[1])
toolbox.register("checking", random.uniform, bounds_low[2], bounds_high[2])
toolbox.register("savings", random.uniform, bounds_low[3], bounds_high[3])

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", 
                tools.initCycle, 
                creator.Individual, 
                (toolbox.amount, toolbox.duration, toolbox.checking, toolbox.savings), 
                n=1)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

toolbox.register("evaluate", get_cost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("mate", 
                tools.cxSimulatedBinaryBounded, 
                low=bounds_low,
                up=bounds_high,
                eta=CROWDING_FACTOR)

toolbox.register("mutate", 
                tools.mutPolynomialBounded,
                low=bounds_low,
                up=bounds_high,
                eta=CROWDING_FACTOR, 
                indpb=1.0/len(bounds_high))


# Genetic Algorithm flow:
def main():

    # Genetic Algorithm flow: ---------------------

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, 
                                                      toolbox, 
                                                      cxpb=P_CROSSOVER, 
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS, 
                                                      stats=stats, 
                                                      halloffame=hof, 
                                                      verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    amount, duration, checking, savings = best
    print(f"-- Best solution: Amount = {int(amount)}, Duration = {int(duration)}, checking = {int(checking)}, savings = {int(savings)}")
    print("-- Prediction: is_risk = ", credit_data.is_credit_risk(applicant.with_values(best)))

if __name__ == "__main__":
    main()
    