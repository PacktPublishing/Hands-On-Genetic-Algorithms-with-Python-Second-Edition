from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import FeatureUnion
import random
import numpy as np

class NewsgroupClassifier:

    def __init__(self, randomSeed):
        random.seed = randomSeed
        self.init_data()

    def __len__(self):
        # total number of features:
        return self.X_train.shape[1]

    def init_data(self):
        print("Initializing newsgroup data...")

        categories = ['rec.autos', 'rec.motorcycles']
        remove = ('headers', 'footers', 'quotes')
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove, shuffle=False)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove, shuffle=False)

        word_vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english",
                                          ngram_range=(1, 3))
        char_vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, max_df=0.5, min_df=5, ngram_range=(2, 5))
        vectorizer = FeatureUnion([('word_vectorizer', word_vectorizer), ('char_vectorizer', char_vectorizer)])
        #vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english", ngram_range=(1, 3))
    
        self.X_train = vectorizer.fit_transform(newsgroups_train.data)
        self.y_train = newsgroups_train.target

        self.X_test = vectorizer.transform(newsgroups_test.data)
        self.y_test = newsgroups_test.target

        self.features_names = vectorizer.get_feature_names_out()

        print(f"Number of features = {self.X_train.shape[1]}, "
              f"train set size = {self.X_train.shape[0]}, "
              f"test set size = {self.X_test.shape[0]}")

    def get_feature_name(self, index):
        return self.features_names[index]

    def get_f1_score(self, features_indices):
        pred = self.get_predictions(features_indices)
        f1_score = metrics.f1_score(self.y_test, pred, average="macro")
        return f1_score
    
    def get_accuracy(self, features_indices):
        pred = self.get_predictions(features_indices)
        accuracy = metrics.accuracy_score(self.y_test, pred)
        return accuracy
    
    def get_predictions(self, features_indices):
        # drop the dataset columns that correspond to the unselected features:
        reduced_X_train = self.X_train[:, features_indices]
        reduced_X_test = self.X_test[:, features_indices]

        classifier = MultinomialNB(alpha=.01)
        classifier.fit(reduced_X_train, self.y_train)
        return classifier.predict(reduced_X_test)
        

def main():

    RANDOM_SEED = 42
    SUBSET_SIZE = 100

    random.seed = RANDOM_SEED

    ngc = NewsgroupClassifier(RANDOM_SEED)

    all_features = range(len(ngc))
    print(f"f1 score using all features: {ngc.get_f1_score(all_features)}")

    random_list = random.sample(range(len(ngc)), SUBSET_SIZE)
    print(f"f1 score using random subset of {SUBSET_SIZE} features: {ngc.get_f1_score(random_list)}")

if __name__ == '__main__':
    main()
