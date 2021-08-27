from utilities import *
from lstm_bidirectional import *


def train_loop(X, trials, loo='hand'):
    batch_size = 128 * 2
    subjs = X.subjID.nunique()

    lrs = np.array([1e-04, 1e-03, 5e-04])
    regs = np.array([1e-04, 1e-03, 5e-04, 5e-03])
    num_layers = np.array([1, 2, 3, 4])
    dropouts = np.arange(start=.1, stop=.5, step=.05)
    hidden_dims = np.array([25, 50, 75, 111])

    combo = np.array(np.meshgrid(lrs, regs, num_layers, dropouts, hidden_dims)).T.reshape(-1, 5)
    scores = np.zeros((trials, subjs))
    true_labels = [[]] * subjs
    pred_labels = [[]] * subjs
    pred_probas = [[]] * subjs

    for t in range(trials):
        lo_out_test = leave_1__out(X, loo)
        for i, (train_, test) in enumerate(lo_out_test):
            if loo == 'hand':
                print(f'{t}) Testing {test.iloc[0, :2].values}')
            else:
                print(f'{t}) Testing {test.iloc[0, :3].values}')

            test_loader = load_loader(test, batch_size=batch_size, train=False)
            tested_index = int(test.iloc[0, 0])
            lo_out_val = leave_1__out(train_, loo)
            running_validated = np.zeros((len(combo)))
            running_tested = np.zeros((len(combo)))
            comodo_pred_labels = [[]] * len(combo)
            comodo_pred_probas = [[]] * len(combo)

            for j, (train, val) in enumerate(lo_out_val):
                train_loader = load_loader(train, batch_size=batch_size, train=True)
                val_loader = load_loader(val, batch_size=batch_size, train=False)

                for h, (lr, reg, num_layer, drop, hidden_dim) in enumerate(combo):
                    print(f'\t-> Filters [{h + 1}/{len(combo)}]')
                    model = LSTM(input_dim=2, hidden_dim=int(hidden_dim), num_layers=int(num_layer), output_size=subjs, drop=drop).to(device)
                    val_acc, test_acc, labels_predicted, proba_predicted = train_model(model, train_loader, val_loader, test_loader, epochs=200, lr=lr, reg=reg)
                    print(f'Testing {i}, Predicted acc --> {test_acc}')
                    running_validated[h] += val_acc
                    running_tested[h] += test_acc

                    if len(comodo_pred_labels[h]) == 0:
                        comodo_pred_labels[h] = labels_predicted
                        comodo_pred_probas[h] = proba_predicted
                    else:
                        comodo_pred_labels[h].extend(labels_predicted)
                        comodo_pred_probas[h].extend(proba_predicted)

            best_hyper = np.argmax(running_validated)
            scores[t, tested_index] += running_tested[best_hyper]
            true_labels[tested_index] += test.iloc[:, 0].values.tolist()
            pred_labels[tested_index] += comodo_pred_labels[best_hyper]
            pred_probas[tested_index] += comodo_pred_probas[best_hyper]

            print(f'final acc on {i}-> {scores[t, tested_index]}')
            pd.DataFrame(scores).to_csv(path + '/lstm/acc_' + loo + '.csv', sep=';', decimal=',', index=False,
                                        header=False)
            pd.DataFrame(true_labels).to_csv(path + '/lstm/true_lab_' + loo + '.csv', sep=';', decimal=',', index=False,
                                             header=False)
            pd.DataFrame(pred_labels).to_csv(path + '/lstm/pred_lab_' + loo + '.csv', sep=';', decimal=',', index=False,
                                             header=False)
            pd.DataFrame(pred_probas).to_csv(path + '/lstm/pred_proba_' + loo + '.csv', sep=';', decimal=',',
                                             index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lstm')
    parser.add_argument('--loo', type=str, default='hand')
    parser.add_argument('--trial', type=int, default=30)
    args = parser.parse_args()

    X = pd.read_csv(path + '/data14.csv', sep=';', decimal='.', header=0)
    train_loop(X, args.trial, args.loo)
