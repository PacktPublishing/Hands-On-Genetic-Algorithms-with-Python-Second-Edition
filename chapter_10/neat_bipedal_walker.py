import gymnasium as gym
import time
import os

import numpy as np
import pickle

from sklearn.neural_network import MLPRegressor

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

INPUTS = 4
OUTPUTS = 1

DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "cart-pole-data.pickle")


class BipedalWalker:

    def __init__(self, randomSeed=None, renderMode=None):

        self.env = gym.make('BipedalWalker-v3', render_mode=renderMode)
        self.randomSeed = randomSeed


    def getScore(self, net):
        """
        calculates the score of a given solution, represented by the list of float-valued network parameters,
        by creating a corresponding MLP Regressor, initiating an episode of the Cart-Pole environment and
        running it with the MLP controlling the actions, while using the observations as inputs.
        Higher score is better.
        :param netParams: a list of floats representing the network parameters (weights and biases) of the MLP
        :return: the calculated score value
        """


        self.env.reset(seed = self.randomSeed)

        actionCounter = 0
        totalReward = 0
        observation, info = self.env.reset(seed = self.randomSeed)
        action = int(net.activate(observation))

        while True:
            actionCounter += 1
            observation, reward, terminated, truncated, info = self.env.step(action)
            totalReward += reward

            if terminated or truncated:
                break
            else:
                action = int(net.activate(observation))
                #print(action)

        return totalReward

    def saveParams(self, net):
        """
        serializes and saves a list of network parameters using pickle
        :param netParams: a list of floats representing the network parameters (weights and biases) of the MLP
        """
        #savedParams = []
        #for param in netParams:
        #    savedParams.append(param)

        pickle.dump(net, open(DATA_FILE_PATH, "wb"))
        
    def replayWithSavedParams(self):
        """
        deserializes a saved list of network parameters and uses it to replay an episode
        """
        savedNet = pickle.load(open(DATA_FILE_PATH, "rb"))
        self.replay(savedNet)

    def replay(self, net):
        """
        renders the environment and uses the given network parameters to replay an episode, to visualize a given solution
        :param netParams: a list of floats representing the network parameters (weights and biases) of the MLP
        """

        # start a new episode:
        actionCounter = 0
        totalReward = 0
        observation, info = self.env.reset(seed = self.randomSeed)
        action = int(net.activate(observation))

        # start rendering:
        self.env.render()

        while True:
            actionCounter += 1
            self.env.render()
            observation, reward, terminated, truncated, info = self.env.step(action)
            totalReward += reward

            print(actionCounter, ": --------------------------")
            print("action = ", action)
            print("observation = ", observation)
            print("reward = ", reward)
            print("totalReward = ", totalReward)
            print("terminated = ", terminated)
            print("truncated = ", truncated)
            print()

            if terminated or truncated:
                break
            else:
                time.sleep(0.03)
                action = int(net.activate(observation))

        self.env.close()


def main():
    cart = BipedalWalker(renderMode='human')
    cart.replayWithSavedParams()

    exit()


if __name__ == '__main__':
    main()