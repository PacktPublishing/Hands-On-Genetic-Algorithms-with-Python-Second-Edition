import time

from flask import Flask
from waitress import serve

app = Flask(__name__)

DELAY_SECONDS = 3

@app.route("/")
def welcome():
    return "<p>Welcome to our Fitness Evaluation Server!</p>"

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

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000, threads=20)