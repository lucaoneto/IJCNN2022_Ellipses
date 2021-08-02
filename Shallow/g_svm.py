from utils import *

def training_loop(X, mc, loo, splits):
    param_grid = {
        'model__C': [1e-03, 1e-02, 1e-01, 1, 10],
        'model__gamma': ['scale', 'auto']
    }
    labels = []
    predicted = []
    predicted_probas = []
    accuracies = np.zeros((mc, X.subjID.nunique()))
    
    for m in range(mc):
        cv = leave_1_out(X, loo)
        running_acc = 0.0
        for i, (train_idx, test_idx) in enumerate(cv):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            subjTested = int(X_test.iloc[0,0])
            
            X_train.reset_index(inplace=True,drop=True)
            cv_Val = leave_1_out(X_train,loo)
            
            model = SVC(kernel='rbf')#, probability=True)
            pipeline = Pipeline(steps = [
                                    ['norm', StandardScaler()],
                                    ['model', model]]
                                   )
            clf = GridSearchCV(pipeline, param_grid, cv=cv_Val, scoring=('accuracy'), n_jobs=32)
            clf.fit(X_train.iloc[:,5:], X_train.subjID)
            y_pred = clf.best_estimator_.predict(X_test.iloc[:,5:])
            
            current_acc = accuracy_score(X_test.subjID, y_pred)
            accuracies[m,subjTested] += current_acc
            running_acc += current_acc
            labels.extend(X_test.subjID.tolist())
            predicted.extend(y_pred.tolist())
            #predicted_probas.extend(clf.best_estimator_.predict_proba(X_test.iloc[:,5:]).tolist())
            
            print(f'repetition {m}/{mc} subj tested = {subjTested} with acc -> {current_acc} \t {running_acc/(i+1)}')
            
            pd.DataFrame(accuracies).to_csv('F:/Damato/Ferrara/cogn/GSVM/accuracies'+loo+splits +'.csv', sep=';', decimal=',', index=False, header=False) 
            pd.DataFrame(labels).to_csv('F:/Damato/Ferrara/cogn/GSVM/labels_'+loo+splits +'.csv', sep=';', decimal=',', index=False, header=False) 
            pd.DataFrame(predicted).to_csv('F:/Damato/Ferrara/cogn/GSVM/predicted_'+loo+splits+'.csv', sep=';', decimal=',', index=False, header=False) 
            pd.DataFrame(predicted_probas).to_csv('F:/Damato/Ferrara/cogn/GSVM/predicted_probas_'+loo+splits +'.csv', sep=';', decimal=',', index=False, header=False) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svm')
    parser.add_argument('--loo', type=str, default='trial',help='leave 1-*-out criterion: trial/speed/inclination')
    parser.add_argument('--mc',type=int,default=100,help='montecarlo steps')
    parser.add_argument('--splits',type=str,default=6,help='montecarlo steps')
    args = parser.parse_args()
    

    print(args.splits)
    
    X = pd.read_csv('F:/Damato/Ferrara/cogn/Data/feat_'+args.splits+'.csv', sep=';', decimal='.',header=0)
    training_loop(X,args.mc,args.loo,args.splits) 