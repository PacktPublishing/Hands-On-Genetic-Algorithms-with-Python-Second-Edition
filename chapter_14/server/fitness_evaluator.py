from flask import Flask

import time

app = Flask(__name__)

DELAY_SECONDS = 3

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

def busy_wait(duration):
    current_time = time.time()
    while (time.time() < current_time + duration):
        pass
               
@app.route("/one_max_fitness/<individual_as_string>")
# fitness calculation:
# compute the number of '1's in the individual
def oneMaxFitness(individual_as_string):
    busy_wait(DELAY_SECONDS)
    individual = [int(char) for char in individual_as_string]
    return str(sum(individual)) # return an str(int)