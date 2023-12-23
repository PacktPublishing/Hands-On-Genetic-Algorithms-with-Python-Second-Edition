import random
import asyncio
from deap import base, creator, tools, algorithms

# Define the evaluation function (calls a web server asynchronously to get the fitness)
async def eval_func(individual):
    # Your asynchronous evaluation logic here
    # ...

# Rest of the code remains unchanged

def main():
    # Create an initial population
    population = toolbox.population(n=10)

    # Set the number of generations
    generations = 5

    # Iterate through generations
    for gen in range(generations):
        # Asynchronously evaluate the population
        tasks = [eval_func(ind) for ind in population]
        fitnesses = asyncio.run(asyncio.gather(*tasks))

        # Assign fitness values to individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select and evolve the population
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring asynchronously
        tasks = [eval_func(ind) for ind in offspring]
        fitnesses = asyncio.run(asyncio.gather(*tasks))

        # Assign fitness values to offspring
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the offspring
        population[:] = offspring

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
