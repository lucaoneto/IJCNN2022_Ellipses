import argparse
import numpy as np
import pandas as pd
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import _LRScheduler


path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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
        feat = g.iloc[:, 7:9].values
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


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def sampleSameFrequency(X):
    subjs = X.subjID.unique()
    n = [X[X.subjID == i].groupby([X.cond1, X.cond2, X.trial, X.nellipse]).ngroups for i in subjs]
    min_groups = min(i for i in n if i > 0)
    ret = []

    for i, s in enumerate(subjs):
        group = X[X.subjID == s].groupby([X.cond1, X.cond2, X.trial, X.nellipse])
        d = group.indices.keys()
        d = random.sample(list(d), min_groups)
        for el in d:
            ret.append(group.get_group(el).values)

    ret = np.asarray(np.vstack(ret))
    ret = pd.DataFrame(ret, columns=X.columns)
    return ret


def train_model(model, trn_dl, val_dl, test_dl, epochs, lr, reg):
    iterations_per_epoch = len(trn_dl)
    best_acc = 0
    test_acc_saved = 0
    patience, trials = 100, 0
    probas_predicted = []
    labels_predicted = []

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))
    print('Start model training')

    for epoch in range(1, epochs + 1):
        model.train()
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            opt.zero_grad()
            out = model(x_batch)

            loss = criterion(out, y_batch)
            l1_regularization = 0.
            for name, param in model.linear.named_parameters():
                if 'weight' in name:
                    l1_regularization += param.abs().sum()

            loss += (reg * l1_regularization.item())
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        correct, total = 0, 0
        for x_tr, y_tr in trn_dl:
            out = model(x_tr)
            preds = F.softmax(out, dim=1).argmax(dim=1)
            total += y_tr.size(0)
            correct += (preds == y_tr).sum().item()

        train_acc = correct / total

        correct, total = 0, 0
        for x_val, y_val in val_dl:
            out = model(x_val)
            preds = F.softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

        val_acc = correct / total

        correct_test, total_test = 0, 0
        pred_probas = []
        pred_labels = []
        for x_test, y_test in test_dl:
            out = model(x_test)
            proba = F.softmax(out, dim=1)
            pred_probas += proba
            preds_test = proba.argmax(dim=1)
            total_test += y_test.size(0)
            pred_labels += preds_test.tolist()
            correct_test += (preds_test == y_test).sum().item()

        test_acc = correct_test / total_test

        if epoch % 5 == 0:
            print(
                f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Train Acc.: {train_acc:2.2%}  Val Acc.: {val_acc:2.2%}  Test Acc.: {test_acc:2.2%}')

        if val_acc > best_acc and epoch > 10:
            trials = 0
            best_acc = val_acc
            test_acc_saved = test_acc
            probas_predicted = pred_probas
            labels_predicted = pred_labels
            print(
                f'Epoch {epoch} best model saved with Loss: {loss.item():.4f}. accuracy: {best_acc:2.2%} Test Acc.: {test_acc_saved:2.2%}')
            if val_acc == 1:
                break
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break

    return best_acc, test_acc_saved, labels_predicted, probas_predicted
