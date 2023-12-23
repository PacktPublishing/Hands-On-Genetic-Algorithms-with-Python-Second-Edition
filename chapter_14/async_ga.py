import random
import asyncio
from deap import base, creator, tools, algorithms

# Define the evaluation function (calls a web server asynchronously to get the fitness)
async def eval_func(individual):
    # Your asynchronous evaluation logic here
    # ...

# Rest of the code remains unchanged

async def evaluate_population_async(population):
    tasks = [eval_func(ind) for ind in population]
    return await asyncio.gather(*tasks)

def main():
    # Create an initial population
    population = toolbox.population(n=10)

    # Set the number of generations
    generations = 5

    # Iterate through generations
    for gen in range(generations):
        # Asynchronously evaluate the population
        fitnesses = asyncio.run(evaluate_population_async(population))

        # Assign fitness values to individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Your crossover, mutation, and selection operations here...

        # Gather fitness data for statistics
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print(f"Generation {gen}: Max Fitness = {max(fits)}, Avg Fitness = {mean}, Std Dev = {std}")

    # Print the final population
    print("\nFinal Population:")
    for ind in population:
        print(f"Individual: {ind}, Fitness: {ind.fitness.values[0]}")

if __name__ == "__main__":
    main()
