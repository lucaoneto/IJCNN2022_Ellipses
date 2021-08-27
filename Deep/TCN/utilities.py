import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, weight_norm

from sklearn import preprocessing
import copy
import os
from functools import partial
from collections import defaultdict


path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def sampleSameFrequency(X):
    subjs = X.subjID.unique()
    n = [X[X.subjID == i].groupby([X.cond1, X.cond2, X.trial, X.nellipse]).ngroups for i in subjs]

    min_groups = min(i for i in n if i > 0)
    ret = []

    for i, _ in enumerate(subjs):
        group = X[X.subjID == i].groupby([X.cond1, X.cond2, X.trial, X.nellipse])

        if n[i] > min_groups:
            d = group.indices.keys()
            d = random.sample(list(d), min_groups)
            for el in d:
                ret.append(group.get_group(el).values)
        else:
            g = group.apply(lambda x: x).values
            if len(g) > 0:
                ret.append(g)

    ret = np.asarray(np.vstack(ret))
    ret = pd.DataFrame(ret, columns=X.columns)
    return ret


def normalization(data, loo):
    if loo == 'speed':
        ret = data.groupby([data.iloc[:, 0], data.iloc[:, 2]]).transform(lambda x: (x - x.mean()) / x.std())
    else:
        ret = data.groupby([data.iloc[:, 0]]).transform(lambda x: (x - x.mean()) / x.std())
    return ret


def normalizeEachEllipse(X):
    groups = X.groupby([X.subjID, X.cond1, X.cond2, X.trial, X.nellipse])
    ret = []

    for name, g in groups:
        s = g.iloc[:, 7:10]
        min_max_scaler_v, min_max_scaler_p, min_max_scaler_r = \
            preprocessing.RobustScaler(), preprocessing.RobustScaler(), preprocessing.RobustScaler()
        v = min_max_scaler_v.fit_transform(s.iloc[:, 0].values.reshape(-1, 1))
        p = min_max_scaler_p.fit_transform(s.iloc[:, 1].values.reshape(-1, 1))
        r = min_max_scaler_p.fit_transform(s.iloc[:, 2].values.reshape(-1, 1))

        new_values = g.values

        new_values[:, 7] = v.flatten()
        new_values[:, 8] = p.flatten()
        new_values[:, 9] = r.flatten()
        ret.append(new_values)

    ret = pd.DataFrame(np.asarray(np.vstack(ret)))
    ret.columns = X.columns
    return X


def leave_1__out(X, loo):
    if loo == 'hand':
        grouped = X.groupby([X.subjID, X.cond1])
    elif loo == 'speed':
        grouped = X.groupby([X.subjID, X.cond1, X.cond2])
    else:
        print('Leave One out not supported')
        return -1

    for i, (name, group) in enumerate(grouped):
        test = group.copy()
        train = X.drop(test.index)
        train = sampleSameFrequency(train)
        yield train, test


def prepareDataToTensor(X):
    groups = X.groupby([X.subjID, X.cond1, X.cond2, X.trial, X.nellipse])
    features, labels = [], []

    for i, (name, g) in enumerate(groups):
        feat = g.iloc[:, 7:9].T.values
        lab = g.iloc[0, 0]
        features.append(feat)
        labels.append(lab)

    features = np.asarray(features)
    labels = np.asarray(labels)
    return torch.from_numpy(features).float().cuda(), torch.from_numpy(labels).long().cuda()


def load_loader(X, batch_size=1024, train=False):
    x, y = prepareDataToTensor(X)
    _dataset = TensorDataset(x, y)

    if not train:
        _loader = DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=False)
    else:
        _loader = DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=True)
    return _loader


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        m.weight.data.normal_(0, 0.01)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)

    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight, mean=1, std=0.01)
        nn.init.constant_(m.bias, 0)


def cosine(epoch, t_max, ampl):
    """Shifted and scaled cosine function."""
    t = epoch % t_max
    return (1 + np.cos(np.pi * t / t_max)) * ampl / 2


def inv_cosine(epoch, t_max, ampl):
    """A cosine function reflected on X-axis."""
    return 1 - cosine(epoch, t_max, ampl)


def one_cycle(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
    """A combined schedule with two cosine half-waves."""
    pct = epoch / t_max
    if pct < pivot:
        return inv_cosine(epoch, pivot * t_max, a1)
    return cosine(epoch - pivot * t_max, (1 - pivot) * t_max, a2)


class Scheduler:
    """Updates optimizer's learning rates using provided scheduling function."""

    def __init__(self, opt, schedule):
        self.opt = opt
        self.schedule = schedule
        self.history = defaultdict(list)

    def step(self, t):
        for i, group in enumerate(self.opt.param_groups):
            lr = self.opt.defaults['lr'] * self.schedule(t)
            group['lr'] = lr
            self.history[i].append(lr)


def train_autoencoder(model, train_loader, val_loader, test_loader, epochs=1000, lr=1e-02):
    iterations_per_epoch = len(train_loader)
    iteration = 0
    period = epochs * iterations_per_epoch
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(parameters, lr=lr)
    sched = Scheduler(opt, schedule=partial(one_cycle, t_max=period, pivot=0.3))

    best_loss = np.Inf
    patience, trials = 10, 0
    criterion = nn.SmoothL1Loss()

    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        true_lab = []
        preds = []

        for j, (x, _) in enumerate(train_loader):
            model.train()
            x = x.to(device)
            x_pred = model(x).to(device)
            opt.zero_grad()
            loss = criterion(x_pred, x)
            loss.backward()
            opt.step()
            sched.step(iteration)
            sum_loss += loss.item() * x.shape[0]
            total += x.shape[0]
            true_lab.append(x.tolist())
            preds.append(x_pred.tolist())

        val_loss = evaluation_metrics_autoencoder(model, val_loader)

        if i % 10 == 0 or i == epochs - 1:
            print("(%d) train loss %.3f, val loss %.3f" % (i, sum_loss / total, val_loss))

        # early stopping
        if val_loss < best_loss and i > 1:
            trials = 0
            best_loss = val_loss
        elif loss >= best_loss:
            trials += 1
            if trials >= patience:
                break

    test_loss = evaluation_metrics_autoencoder(model, test_loader)
    return best_loss, test_loss, true_lab, preds


def evaluation_metrics_autoencoder(model, valid_dl):
    total = 0
    sum_loss = 0.0
    criterion = nn.SmoothL1Loss()
    with torch.no_grad():
        for k, (x, _) in enumerate(valid_dl):
            x = x.to(device)
            x_hat = model(x).to(device)

            loss = criterion(x_hat, x)
            sum_loss += loss.item() * x.shape[0]
            total += x.shape[0]
    return sum_loss / total


def train_model(model, train_loader, test_loader, epochs=1000, lr=1e-03, reg_fc=1e-04):
    iterations_per_epoch = len(train_loader)
    iteration = 0
    period = epochs * iterations_per_epoch
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(parameters, lr=lr)
    sched = Scheduler(opt, schedule=partial(one_cycle, t_max=period, pivot=0.3))
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0
        for j, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            y_pred = model(x).to(device)

            opt.zero_grad()
            loss = criterion(y_pred, y)

            l1_regularization = 0.
            for name, param in model.fc.named_parameters():
                if 'weight' in name:
                    l1_regularization += param.norm(1)
            loss += l1_regularization.item() * reg_fc

            clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            opt.step()
            sched.step(iteration)
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            _, pred = torch.max(y_pred.data, 1)
            correct += (pred == y.long()).float().sum()

        train_acc = correct / total
        test_loss, test_acc, labels_predicted, pred_probas = evaluation_metrics(model, test_loader)

        if i % 10 == 0 or i == epochs - 1:
            print("(%d) train loss %.3f, train_acc %.2f, test_acc %.2f" % (
                i, sum_loss / total, (train_acc * 100), (test_acc * 100)))

    return train_acc, test_acc, labels_predicted, pred_probas


def evaluation_metrics(model, valid_dl):
    correct = 0
    total = 0
    sum_loss = 0.0
    pred_probas = []
    pred_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for k, (x, y) in enumerate(valid_dl):
            x, y = x.to(device), y.to(device)
            y_hat = model(x).to(device)

            loss = criterion(y_hat, y)
            prob = F.softmax(y_hat, dim=1)
            pred_probas += prob
            preds_test = prob.argmax(dim=1)
            correct += (preds_test == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item() * y.shape[0]

            pred_labels += preds_test.tolist()

    return sum_loss / total, (correct / total).item(), pred_labels, pred_probas
