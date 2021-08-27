from utils import *


def training_loop(X, mc, loo, splits):
    n_features = X.iloc[:, 5:].shape[1]
    sqrt1_3 = round(pow(n_features, 1 / 3))
    sqrt3_4 = round(pow(n_features, 2 / 3))
    params = ['auto', sqrt1_3, sqrt3_4]

    param_grid = {
        'model__max_features': params
    }

    labels = []
    predicted = []
    predicted_probas = []
    accuracies = np.zeros((mc, X.subjID.nunique()))

    for m in range(mc):
        cv = leave_1_out(X, loo)
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            subjTested = int(X_test.iloc[0, 0])

            X_train.reset_index(inplace=True, drop=True)
            cv_Val = leave_1_out(X_train, loo)

            model = RandomForestClassifier(n_estimators=1000)
            pipeline = Pipeline(steps=[['model', model]]
                                )
            clf = GridSearchCV(pipeline, param_grid, cv=cv_Val, scoring=('accuracy'), n_jobs=32)
            clf.fit(X_train.iloc[:, 5:], X_train.subjID)
            y_pred = clf.best_estimator_.predict(X_test.iloc[:, 5:])

            current_acc = accuracy_score(X_test.subjID, y_pred)
            accuracies[m, subjTested] += current_acc
            labels.extend(X_test.subjID.tolist())
            predicted.extend(y_pred.tolist())
            predicted_probas.extend(clf.best_estimator_.predict_proba(X_test.iloc[:, 5:]).tolist())

            print(f'repetition {m}/{mc} subj tested = {subjTested} with acc -> {current_acc}')

            pd.DataFrame(accuracies).to_csv('RF/accuracies' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                            header=False)
            pd.DataFrame(labels).to_csv('RF/labels_' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                        header=False)
            pd.DataFrame(predicted).to_csv('RF/predicted_' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                           header=False)
            pd.DataFrame(predicted_probas).to_csv('RF/predicted_probas_' + loo + splits + '.csv', sep=';', decimal=',',
                                                  index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rf')
    parser.add_argument('--loo', type=str, default='trial', help='leave 1-*-out criterion: hand/speed')
    parser.add_argument('--mc', type=int, default=30, help='montecarlo steps')
    parser.add_argument('--splits', type=str, default=6, help='splits to be considered')
    args = parser.parse_args()

    X = pd.read_csv('data/feat_' + args.splits + '.csv', sep=';', decimal='.', header=0)
    training_loop(X, args.mc, args.loo, args.splits)
