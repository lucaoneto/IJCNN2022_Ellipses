from utils import *


def training_loop(X, mc, loo, splits):
    param_grid = {
        'model__C': [1e-03, 1e-02, 1e-01, 1, 10],
        'model__gamma': ['scale', 'auto']
    }
    labels = []
    predicted = []
    accuracies = np.zeros((mc, X.subjID.nunique()))

    for m in range(mc):
        cv = leave_1_out(X, loo)
        running_acc = 0.0
        for i, (train_idx, test_idx) in enumerate(cv):
            x_train = X.iloc[train_idx]
            x_test = X.iloc[test_idx]
            subj_tested = int(x_test.iloc[0, 0])

            x_train.reset_index(inplace=True, drop=True)
            cv_val = leave_1_out(x_train, loo)

            model = SVC(kernel='rbf')
            pipeline = Pipeline(steps=[
                ['norm', StandardScaler()],
                ['model', model]]
            )
            clf = GridSearchCV(pipeline, param_grid, cv=cv_val, scoring='accuracy', n_jobs=32)
            clf.fit(x_train.iloc[:, 5:], x_train.subjID)
            y_pred = clf.best_estimator_.predict(x_test.iloc[:, 5:])

            current_acc = accuracy_score(x_test.subjID, y_pred)
            accuracies[m, subj_tested] += current_acc
            running_acc += current_acc
            labels.extend(x_test.subjID.tolist())
            predicted.extend(y_pred.tolist())

            print(
                f'repetition {m}/{mc} subj tested = {subj_tested} with acc -> {current_acc} \t {running_acc / (i + 1)}')

            pd.DataFrame(accuracies).to_csv('GSVM/accuracies' + loo + splits + '.csv', sep=';', decimal=',',
                                            index=False, header=False)
            pd.DataFrame(labels).to_csv('GSVM/labels_' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                        header=False)
            pd.DataFrame(predicted).to_csv('GSVM/predicted_' + loo + splits + '.csv', sep=';', decimal=',', index=False,
                                           header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gaussian_svm')
    parser.add_argument('--loo', type=str, default='trial', help='leave 1-*-out criterion: hand/speed')
    parser.add_argument('--mc', type=int, default=30, help='repetitions')
    parser.add_argument('--splits', type=str, default=6, help='splits to be considered')
    args = parser.parse_args()

    X = pd.read_csv('data/feat_' + args.splits + '.csv', sep=';', decimal='.', header=0)
    training_loop(X, args.mc, args.loo, args.splits)
