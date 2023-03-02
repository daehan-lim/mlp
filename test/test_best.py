import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from util.util import x_y_split
from tabulate import tabulate
import timeit
import random


if __name__ == '__main__':
    dataset = pd.read_csv("../data/dataset_binary.csv")
    random.seed(10)
    indices = list(range(0, len(dataset)))
    random.shuffle(indices)
    training_set = dataset.iloc[indices[460:], :].reset_index(drop=True)
    test_set = dataset.iloc[indices[:460], :].reset_index(drop=True)
    X_train, y_train = x_y_split(training_set)
    X_test, y_test = x_y_split(test_set)

    clf = MLPClassifier(random_state=1, verbose=True, hidden_layer_sizes=(3, 7, 3),
                        )
    clf.fit(X_train, y_train)
    print("\n")
    print(clf)
    print(clf.get_params())
    y_pred = clf.predict(X_test)

    # print(timeit.timeit(lambda: clf.score(X_test, y_test), number=1))
    # print(timeit.timeit(lambda: accuracy_score(y_test, y_pred), number=1))
    # score = clf.score(X_test, y_test)  # takes longer: 0.035 vs 0.0004 accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

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
    print('f1 (on testset): %.3f' % f1)
    # print('roc auc (on testset): %.3f' % roc_auc_score(y_test, y_pred))
    # print('\033[1m' + 'Accuracy: ' + '\033[0m' + str(accuracy))
    print(f"# of iterations: {clf.n_iter_}")
    print(f"Loss: {round(clf.loss_, 3)}")
    print(f"# of coefs: {len(clf.coefs_)}")
    print(f"Name of Output Layer Activation Function: {clf.out_activation_}")
    print("-----------------------------------------------------------------------------")
