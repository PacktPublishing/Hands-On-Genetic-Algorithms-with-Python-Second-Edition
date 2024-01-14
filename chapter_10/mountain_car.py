import gymnasium as gym
import time
import pickle
import os

MAX_STEPS = 200
FLAG_LOCATION = 0.5
DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "mountain-car-data.pickle")


class MountainCar:

    def __init__(self, randomSeed, renderMode=None):

        self.env = gym.make('MountainCar-v0', render_mode=renderMode)
        self.randomSeed = randomSeed

    def __len__(self):
        return MAX_STEPS

    def getScore(self, actions):
        """
        calculates the score of a given solution, represented by the list of actions, for the Moutain-Car environment,
        by initiating an episode of the Mountain-Car environment and running it with the provided actions.
        Lower score is better.
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        :return: the calculated score value
        """

        # start a new episode:
        self.env.reset(seed=self.randomSeed)

        actionCounter = 0

        # feed the actions to the environment:
        for action in actions:
            actionCounter += 1

            # provide an action and get feedback:
            observation, reward, terminated, truncated, info = self.env.step(action)

            # episode over - either the car hit the flag, or 200 actions processed:
            if terminated or truncated:
                break

        # evaluate the results to produce the score:
        if actionCounter < MAX_STEPS:
            # the car hit the flag:
            # start from a score of 0
            # reward further for a smaller amount of steps
            score = 0 - (MAX_STEPS - actionCounter)/MAX_STEPS
        else:
            # the car did not hit the flag:
            # reward according to distance from flag
            score = abs(observation[0] - FLAG_LOCATION)  # we want to minimize that

        return score

    def saveActions(self, actions):
        """
        serializes and saves a list of actions using pickle
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        """
        savedActions = []
        for action in actions:
            savedActions.append(action)

        os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
        pickle.dump(savedActions, open(DATA_FILE_PATH, "wb"))

    def replaySavedActions(self):
        """
        deserializes a saved list of actions and replays it
        """
        savedActions = pickle.load(open(DATA_FILE_PATH, "rb"))
        self.replay(savedActions)

    def replay(self, actions):
        """
        renders the environment and replays list of actions into it, to visualize a given solution
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        """
        # start a new episode:
        observation = self.env.reset()

        # start rendering:
        self.env.render()

        actionCounter = 0

        # replay the given actions by feeding them into the environment:
        for action in actions:

            actionCounter += 1
            self.env.render()
            observation, reward, terminated, truncated, info = self.env.step(action)
            print(actionCounter, ": --------------------------")
            print("action = ", action)
            print("observation = ", observation)
            print("distance from flag = ", abs(observation[0] - 0.5))
            print()

            if terminated or truncated:
                break
            else:
                time.sleep(0.02)

        self.env.close()


def main():

    RANDOM_SEED = 42
    car = MountainCar(RANDOM_SEED, renderMode="human")
    car.replaySavedActions()

if __name__ == '__main__':
    main()