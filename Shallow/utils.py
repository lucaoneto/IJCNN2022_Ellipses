import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def leave_1_out(X, loo):
    if loo != 'hand':
        grouped = X.groupby([X.subjID, X.cond1, X.cond2])
    else:
        grouped = X.groupby([X.subjID, X.cond1])

    for i, (name, group) in enumerate(grouped):
        test = group.copy()
        train = X.drop(test.index)
        mi = min(train.subjID.value_counts())
        train = train.groupby(train.subjID, group_keys=False).apply(lambda x: x.sample(min(len(x), mi)))
        yield train.index, test.index
