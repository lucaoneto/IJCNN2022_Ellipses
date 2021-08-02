import numpy as np
import os,sys
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def leave_1_out(X, loo):
  if loo == 'hand':
    grouped = X.groupby([X.subjID, X.cond1])
  else:
    grouped = X.groupby([X.subjID, X.cond1, X.cond2])
  
  for i, (name,group) in enumerate(grouped):
    test = group.copy()
    train = X.drop(test.index)
    mi = min(train.subjID.value_counts())
    train = train.groupby(train.subjID, group_keys=False).apply(lambda x: x.sample(min(len(x), mi)))
    yield train.index, test.index