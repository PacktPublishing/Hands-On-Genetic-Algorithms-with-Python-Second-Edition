from neatpy.options import Options
from neatpy.population import Population
from neatpy.draw import draw_brain_pygame, draw_species_bar_pygame
from neatpy.save import save_brain
from neatpy.brain import Brain
import pygame as pg
from pygame.color import THECOLORS as colors

import itertools
import random
import os

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

NUM_INPUTS = 3

# create folder if does not exist:
IMAGE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
    
# setup the graphics display:
pg.init()
screen = pg.display.set_mode((400, 400))
screen.fill(colors['lightblue'])

# set num_inputs, num_outputs, population_size, fitness_threshold:
Options.set_options(NUM_INPUTS, 1, 150, 2**NUM_INPUTS - 0.1)

# create even parity truth-table:
parityIn = list(itertools.product([0, 1], repeat=NUM_INPUTS))
parityOut = [sum(row) % 2 for row in parityIn]

# calculate the score based on differences between the 
# generated outputs and the expected parity outputs:
def parityScore(nn):
    score = 2**NUM_INPUTS 

    for pIn, pOut in zip(parityIn, parityOut):
        output = nn.predict(pIn)[0]
        score-= (output - pOut) ** 2


    # add a small penalty for network size:
    score -= len(nn.nodes) * 0.01
    
    return score

# draw current best network and current species:
def draw_current(population):
    draw_brain_pygame(screen, population.best, dim=250, x=75,
                      y=120, circle_size=10, line_width=3)
    
    draw_species_bar_pygame(screen, population, x=0, y=0, 
                            width=400, height=100)
    
    pg.display.update()

# NEAT main loop: --------

p = Population()

import time
while p.best.fitness < Options.fitness_threshold: 

    for nn in p.pool:
        nn.fitness = parityScore(nn)

    p.epoch()

    print(p)
    draw_current(p)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            quit()

# print results:
print(f"best network = {p.best.get_draw_info()}")
print(f"best fitness = {p.best.fitness}")
print(f"Number of nodes = {len(p.best.nodes)}")

print("Checking the truth table:")
for pIn, pOut in zip(parityIn, parityOut):
    output = p.best.predict(pIn)[0]
    print(f"input {pIn}, expected output {pOut}, got {output:.3f} -> {round(output)}")


# save results:
save_brain(p.best, 'brain_test')
pg.image.save(screen, os.path.join(IMAGE_PATH, f'even-parity-{NUM_INPUTS}.png'))