from embeddings import Embeddings
import numpy as np

MODEL =  "glove-twitter-50" 

class MysteryWordGame:

    def __init__(self, given_mystery_word=None):

        self.embeddings = Embeddings(model_name=MODEL)

        self.mystery_word = given_mystery_word if given_mystery_word else self.embeddings.pick_random_embedding()
        print(f"--- Mistery word is '{self.mystery_word}' — game on!")
    
    def score_guess(self, guess_word):
        if self.embeddings.has_word(guess_word):
             score = 100 * self.embeddings.get_similarity(self.mystery_word, guess_word)
        else:
            score = -100
        return score
            

def main():
    
    game = MysteryWordGame(given_mystery_word="dog")

    print("-- Checking candidate guess words for 'dog':")

    for guess_word in ["computer", "asdghf", "canine", "hound", "poodle", "puppy", "cat", "dog"]:
        score = game.score_guess(guess_word)
        print(f"- current guess: {guess_word.ljust(10)} => score = {score:.2f}")

if __name__ == '__main__':
    main()
