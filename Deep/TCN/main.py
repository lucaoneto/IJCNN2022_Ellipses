from utilities import *
from models import *


def train_loop(X, trials, loo='hand'):
    batch_size = 256
    subjs = X.subjID.nunique()
    lrs = np.array([1e-04, 5e-04, 1e-03, 5e-03])
    l1s = np.array([1e-02, 5e-03])

    num_layers = np.array([1, 2, 3])
    dropouts = np.arange(start=.1, stop=.5, step=.05)
    num_filters = np.array([32, 64, 128])
    combo = np.array(np.meshgrid(lrs, l1s, num_layers, num_filters, dropouts)).T.reshape(-1, 5)

    scores = np.zeros((trials, subjs))
    true_labels = [[]] * subjs
    pred_labels = [[]] * subjs

    for t in range(trials):
        lo_out_test = leave_1__out(X, loo)
        for i, (train_, test) in enumerate(lo_out_test):
            if loo == 'hand':
                print(f'{t}) Testing {test.iloc[0, :2].values}')
            else:
                print(f'{t}) Testing {test.iloc[0, :3].values}')
            test_loader = load_loader(test, batch_size=batch_size)
            tested_index = int(test.iloc[0, 0])
            lo_out_val = leave_1__out(train_, loo)

            running_validated = np.zeros((len(combo)))
            running_tested = np.zeros((len(combo)))
            comodo_true_labels = [[]] * len(combo)
            comodo_pred_labels = [[]] * len(combo)
            n_validation = 0

            for j, (train, val) in enumerate(lo_out_val):
                train_loader = load_loader(train, batch_size=batch_size)
                val_loader = load_loader(val, batch_size=batch_size)
                n_validation += 1

                for h, (lr, l1, num_layers, num_filters, drop) in enumerate(combo):
                    print(f'\t-> Filters [{h + 1}/{len(combo)}]')
                    autoencoder = TCN_auto(num_layers=num_layers, num_filters=num_filters, kernel_sizes=[5, 7], dropout=drop).to(device)
                    autoencoder.apply(weights_init)
                    val_loss, _, _, _ = \
                        train_autoencoder(autoencoder, train_loader, val_loader, test_loader, epochs=100, lr=lr)

                    cnn = TCN(num_layers=num_layers, num_filters=num_filters, kernel_sizes=[5, 7], dropout=drop).to(device)
                    cnn.feature1 = copy.deepcopy(autoencoder.tcn.feature1)
                    cnn.feature2 = copy.deepcopy(autoencoder.tcn.feature2)
                    cnn.feature3 = copy.deepcopy(autoencoder.tcn.feature3)
                    best_loss, test_acc, preds, labels = \
                        train_model(cnn, train_loader, val_loader, test_loader, reg_fc=l1)

                    print(f'Predicted acc --> {test_acc}')
                    running_validated[h] += best_loss
                    running_tested[h] += test_acc

                    if len(comodo_pred_labels[h]) == 0:
                        comodo_pred_labels[h] = preds
                        comodo_true_labels[h] = labels
                    else:
                        comodo_pred_labels[h].extend(preds)
                        comodo_true_labels[h].extend(labels)

            best_hyper = np.argmin(running_validated)
            scores[t, tested_index] += running_tested[best_hyper] / n_validation
            true_labels[tested_index] += comodo_true_labels
            pred_labels[tested_index] += comodo_pred_labels[best_hyper]

            print(f'final acc -> {scores[t, tested_index]}')
            pd.DataFrame(scores).to_csv(path + '/tcn/acc_' + loo + '.csv', sep=';', decimal=',', index=False,
                                        header=False)
            pd.DataFrame(true_labels).to_csv(path + '/tcn/true_lab_' + loo + '.csv', sep=';', decimal=',', index=False,
                                             header=False)
            pd.DataFrame(pred_labels).to_csv(path + '/tcn/pred_lab_' + loo + '.csv', sep=';', decimal=',', index=False,
                                             header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tcn')
    parser.add_argument('--loo', type=str, default='hand')
    parser.add_argument('--mc', type=int, default=30)
    args = parser.parse_args()

    X = pd.read_csv(path + '/data14.csv', sep=';', decimal='.', header=0)
    train_loop(X, args.mc, args.loo)
