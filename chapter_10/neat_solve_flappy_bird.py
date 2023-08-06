import os
import random
import neat_flappy_bird as flappy_bird

import neat
import neat_visualize as visualize

POPULATION_SIZE = 300

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the cart pole task class:
flappyBird = flappy_bird.FlappyBird(RANDOM_SEED)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #genome.fitness = flappyBird.getScore(net)
        
        total = 0
        for i in range(10):
            total = total + flappyBird.getScore(net)
        genome.fitness = total / 10.0
        


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, POPULATION_SIZE) #TODO

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_fitness = flappyBird.getScore(winner_net)
    print(f"best fitness = {winner_fitness}")

    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    flappyBird.saveParams(best_net)

    #node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'action'}
    visualize.draw_net(config, winner, True)  #, node_names=node_names)
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 100)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.properties')
    run(config_path)
