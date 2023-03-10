import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from util.util import x_y_split
from tabulate import tabulate
from util.util import timeit


@timeit
def main():
    training_set = pd.read_csv("data/training_set.csv")
    test_set = pd.read_csv("data/test_set.csv")
    X_train, y_train = x_y_split(training_set)
    X_test, y_test = x_y_split(test_set)

    params = {#'activation': ['relu', 'tanh', 'logistic'],
              'hidden_layer_sizes': [(3, 7, 3)],
              'max_iter': [50, 200, 400],
              # 'solver': ['adam', 'sgd',],
              # 'alpha': [0.0001, 0.001, 0.01, 0.05],
              # 'learning_rate': ['constant', 'adaptive',]
              }

    clf = MLPClassifier(random_state=1)
    clf_grid = GridSearchCV(clf, param_grid=params, n_jobs=-1, scoring='f1', verbose=True, cv=5)
    clf_grid.fit(X_train, y_train)

    print("")
    print(clf_grid)
    print(f"cv={clf_grid.cv}, max_iter = {clf.max_iter}")

    print(f"\nBest params: {clf_grid.best_params_}")
    # print(f"Best estimator: {clf_grid.best_estimator_}")

    # means = clf_grid.cv_results_['mean_test_score']
    # stds = clf_grid.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf_grid.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print(f"Best f1: {clf_grid.best_score_}")
    y_pred = clf_grid.predict(X_test)
    print('Test f1 (on testset): %.3f' % f1_score(y_test, y_pred))
    # print('Test f1 (on testset): %.3f' % clf_grid.score(X_test, y_test)) #takes longer than above

    # print(timeit.timeit(lambda: clf.score(X_test, y_test), number=1))
    # print(timeit.timeit(lambda: accuracy_score(y_test, y_pred), number=1))
    # score = clf.score(X_test, y_test)  # takes longer: 0.035 vs 0.0004 accuracy
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
    print("-----------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
