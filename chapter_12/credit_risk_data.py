from copy import deepcopy
import json
import os
import random
import pandas as pd
import enum

from sklearn.impute import SimpleImputer
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from typing import NamedTuple, Literal 
from dataclasses import dataclass, asdict, field, InitVar
from sklearn.preprocessing import OneHotEncoder

from joblib import dump, load

from applicant import Applicant

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "credit_risk_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "model.joblib")

NUM_FOLDS = 5


class CreditRiskData:
    """This class encapsulates the German Credit Risk classification problem 
    """

    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed

        self.dataset = self.read_dataset()

        # model already exists -> read from file:
        if os.path.isfile(MODEL_PATH):          
            self.classifier = load(MODEL_PATH) 

        # no model yet -> train a new one and save to file:
        else:
            self.classifier = self.train_model()
            dump(self.classifier, MODEL_PATH) 

    def read_dataset(self):
        print("Loading the dataset...")

        # read the dataset from the CSV file:
        df = pd.read_csv(DATASET_PATH)
        
        # replace all categorical variables with 'one-hot' encoding:
        df = pd.get_dummies(df) 

        return df

    def train_model(self):
       
        # separate the features from the result:
        X = self.dataset.drop(columns=['credit_risk'])
        y = self.dataset['credit_risk'].astype(int)
        
        # create a random forest classfier:
        classifier = RandomForestClassifier(random_state=self.randomSeed)

        # split the data into 'folds' for the k-fold validation process:
        kfold = model_selection.KFold(n_splits=NUM_FOLDS)

        # perform k-fold validation and calculate the mean accuracy:
        cv_results = model_selection.cross_val_score(classifier, X, y, cv=kfold, scoring='accuracy')
        print(f"Model's Mean k-fold accuracy = {cv_results.mean()}")

        # train the classifier on the entire dataset:
        classifier.fit(X, y)

        # find the training accuracy of the model:
        y_pred = classifier.predict(X)
        print(f"Model's Training Accuracy = {accuracy_score(y, y_pred)}")

        print("------- Feature Importance values: ")
        feature_importances = dict(zip(X.columns, classifier.feature_importances_))
        # sort the dictionary by the values, in descending order:
        sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: -item[1]))
        print(json.dumps(sorted_feature_importances, indent=4))

        return classifier
        
    def is_credit_risk(self, applicant):
        return self.classifier.predict(applicant.row)[0] == 1

    def risk_probability(self, applicant):
        return self.classifier.predict_proba(applicant.row)[0][1]

    def get_row(self, i):
        return self.dataset.iloc[[i]].drop(columns=['credit_risk'])     

    def get_applicant(self, row_num):
        applicant = Applicant(self.get_row(row_num))
        applicant.pretty_print()
        print(f" => Credit risk = {self.is_credit_risk(applicant)}")
        print()
        return applicant
    

def main():
    credit_data = CreditRiskData(randomSeed=42)

    print("Before modifications: -------------")
    applicant = credit_data.get_applicant(25)

    print("After modifications: -------------")
    modified_applicant = applicant.with_values([1000, 20, 2, 0])
    modified_applicant.pretty_print()
    print(f" => Credit risk = {credit_data.is_credit_risk(modified_applicant)}")
    print()

if __name__ == "__main__":
    main()
