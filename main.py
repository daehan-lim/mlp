import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from tabulate import tabulate
from daehan_mlutil import utilities


@utilities.timeit
def main():
    dataset = pd.read_csv("data/dataset_binary.csv")
    random.seed(10)
    transactions_0 = dataset[dataset['class'] == 0]
    transactions_1 = dataset[dataset['class'] == 1]

    indices = list(range(0, len(transactions_0)))
    random.shuffle(indices)
    test_set_0 = transactions_0.iloc[indices[:417], :].reset_index(drop=True)
    training_set_0 = transactions_0.iloc[indices[417:], :].reset_index(drop=True)

    indices = list(range(0, len(transactions_1)))
    random.shuffle(indices)
    test_set_1 = transactions_1.iloc[indices[:43], :].reset_index(drop=True)
    training_set_1 = transactions_1.iloc[indices[43:], :].reset_index(drop=True)

    training_set = pd.concat([training_set_0, training_set_1])
    test_set = pd.concat([test_set_0, test_set_1])
    X_train, y_train = utilities.x_y_split(training_set, 'class')
    X_test, y_test = utilities.x_y_split(test_set, 'class')

    X, y = utilities.x_y_split(dataset, 'class')
    params = {  # 'activation': ['relu', 'tanh', 'logistic'],
              'hidden_layer_sizes': [(50, 30, 10),
                                     ],
              # 'max_iter': [50, 200, 400],
              # 'solver': ['adam', 'sgd',],
              # 'alpha': [0.0001, 0.001, 0.01],
              # 'learning_rate': ['constant', 'adaptive',]
              }

    clf = MLPClassifier(random_state=1, max_iter=50, activation='logistic', alpha=0.0001)
    clf_grid = GridSearchCV(clf, param_grid=params, n_jobs=-1, scoring=['f1', 'roc_auc'], verbose=True,
                            cv=10, refit='f1')
    clf_grid.fit(X, y)

    print("")
    print(clf_grid)
    print(f"cv={clf_grid.cv}, max_iter = {clf.max_iter}")

    print(f"\nBest params: {clf_grid.best_params_}")

    # means = clf_grid.cv_results_['mean_test_f1']
    # stds = clf_grid.cv_results_['std_test_f1']
    # for mean, std, params in zip(means, stds, clf_grid.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(f"Best f1: {clf_grid.best_score_}")
    print(f"Roc auc on best estimator: {clf_grid.cv_results_['mean_test_roc_auc'][clf_grid.best_index_]}")

    '''
    y_pred = clf_grid.predict(X_test)
    print('Test f1 (on testset): %.3f' % f1_score(y_test, y_pred))
    print('roc auc (on testset): %.3f' % roc_auc_score(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)

    TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))

    confusion_matrix = [
        ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
        ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
         str(TP + FN) + "\n(Total actual c= '1')"],
        ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
         str(FP + TN) + "\n(Total actual c= '0')"],
        ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
         str(len(test_set))],
    ]
    print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
    print(classification_report(y_test, y_pred))
    print('\033[1m' + 'Accuracy: ' + '\033[0m' + str(accuracy))
    # print(f"# of iterations: {clf.n_iter_}")
    # print(f"Loss: {round(clf.loss_, 3)}")
    # print(f"# of coefs: {len(clf.coefs_)}")
    # print(f"Name of Output Layer Activation Function: {clf.out_activation_}")
    '''
    print("-----------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
