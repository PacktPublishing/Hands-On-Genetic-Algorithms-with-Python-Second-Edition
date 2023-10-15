import os
import json
import random

import numpy as np
import gensim.downloader as api

from os.path import join, isfile
from gensim.models import KeyedVectors


MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
DEFAULT_MODEL = 'glove-wiki-gigaword-50'

class Embeddings:

    def __init__(self, model_name=DEFAULT_MODEL, randomSeed=None):

        self.model_name = model_name

        if randomSeed:
            random.seed(randomSeed)
            np.random.seed(randomSeed)

        self._init_model()
        self.words_in_model = list(self.model.index_to_key)

    def _init_model(self):
        model_path = os.path.join(MODEL_PATH, f"{self.model_name}.bin")
        
        if not isfile(model_path):
            self._download_and_save_model(model_path)

        print(f"Loading model '{self.model_name}' from local file...")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def _download_and_save_model(self, model_path):
        print(f"Downloading model {self.model_name}...")
        model = api.load(self.model_name)
        print(f"Saving model to {model_path}...")
        model.save_word2vec_format(model_path, binary=True)

    def get_vector_size(self):
        return self.model.vector_size

    def pick_random_embedding(self):
        random_word = random.choice(self.words_in_model)
        return self.word2vec(random_word)

    def get_similarity(self, word_1, word_2):
        return self.model.similarity(word_1, word_2)

    def vec2_nearest_word(self, vec):
        return self.model.similar_by_vector(vec, topn=1)[0][0]
    
    def has_word(self, word):
        return word in self.model
    
    def word2vec(self, word):
        return self.model.get_vector(word)

    @staticmethod
    def list_models():
        # List available pre-trained models
        info = api.info()

        # Print the list of models
        print("Available Pre-trained Models: -------- ")
        for model_name, model_info in info["models"].items():
            print(f"--- {model_name}: {json.dumps(model_info, sort_keys=True, indent=4)}")
            print()

    
def main():
    embeddings = Embeddings(randomSeed=42)
    embeddings.list_models()

    vec1 = embeddings.word2vec('dog')
    word1 = embeddings.vec2_nearest_word(vec1)
    print(f"{word1} -> {vec1}")

    print()
    print("Random word:")
    vec2 = embeddings.pick_random_embedding()
    word2 = embeddings.vec2_nearest_word(vec2)
    print(f"{word2} -> {vec2}")


if __name__ == '__main__':
    main()

