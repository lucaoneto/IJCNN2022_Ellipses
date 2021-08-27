from utils import *


def training_loop(X, mc, loo, splits):
    param_grid = {
        'model__var_smoothing': [1e-09, 5e-08, 1e-08]
    }
    labels = []
    predicted = []
    predicted_probas = []
    accuracies = np.zeros((mc, X.subjID.nunique()))

    for m in range(mc):
        cv = leave_1_out(X, loo)
        for i, (train_idx, test_idx) in enumerate(cv):
            x_train = X.iloc[train_idx]
            x_test = X.iloc[test_idx]
            subj_tested = int(x_test.iloc[0, 0])

            x_train.reset_index(inplace=True, drop=True)
            cv_val = leave_1_out(x_train, loo)

            model = GaussianNB()
            pipeline = Pipeline(steps=[
                ['norm', StandardScaler()],
                ['model', model]]
            )
            clf = GridSearchCV(pipeline, param_grid, cv=cv_val, scoring='accuracy', n_jobs=32)
            clf.fit(x_train.iloc[:, 5:], x_train.subjID)
            y_pred = clf.best_estimator_.predict(x_test.iloc[:, 5:])

            current_acc = accuracy_score(x_test.subjID, y_pred)
            accuracies[m, subj_tested] += current_acc
            labels.extend(x_test.subjID.tolist())
            predicted.extend(y_pred.tolist())
            predicted_probas.extend(clf.best_estimator_.predict_proba(x_test.iloc[:, 5:]).tolist())

            print(f'repetition {m}/{mc} subj tested = {subj_tested} with acc -> {current_acc}')

            pd.DataFrame(accuracies).to_csv('GaussianNB/accuracies' + loo + splits + '.csv', sep=';', decimal=',',
                                            index=False, header=False)
            pd.DataFrame(labels).to_csv('GaussianNB/labels_' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                        header=False)
            pd.DataFrame(predicted).to_csv('GaussianNB/predicted_' + loo + splits + '.csv', sep=';', decimal=',',
                                           index=False, header=False)
            pd.DataFrame(predicted_probas).to_csv('GaussianNB/predicted_probas_' + loo + splits + '.csv', sep=';',
                                                  decimal=',', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='naive_bayes')
    parser.add_argument('--loo', type=str, default='trial', help='leave 1-*-out criterion: hand/speed')
    parser.add_argument('--mc', type=int, default=30, help='repetitions')
    parser.add_argument('--splits', type=str, default=6, help='splits to be considered')
    args = parser.parse_args()

    X = pd.read_csv('data/feat_' + args.splits + '.csv', sep=';', decimal='.', header=0)
    training_loop(X, args.mc, args.loo, args.splits)
