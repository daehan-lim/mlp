import time

import numpy as np
import pandas as pd
from daehan_mlutil import utilities
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, precision_score, \
    recall_score
from tabulate import tabulate
import random


if __name__ == '__main__':
    start_time = time.time()
    training_set = pd.read_csv("../data/training_set_binary_stand.csv")
    test_set = pd.read_csv("../data/test_set_binary_stand.csv")
    # dataset = pd.read_csv("../data/dataset_binary_all.csv")
    # seeds [0, 10, 35, 42, 123, 456, 789, 101112, 131415, 161718]
    auc_sum = 0
    f1_sum = 0
    precision_sum = 0
    recall_sum = 0
    accuracy_sum = 0
    for seed in range(10):
        print(f"\n\nseed: {seed}")
        random.seed(seed)
        # transactions_0 = dataset[dataset['class'] == 0]
        # transactions_1 = dataset[dataset['class'] == 1]
        #
        # indices = list(range(0, len(transactions_0)))
        # random.shuffle(indices)
        # test_set_0 = transactions_0.iloc[indices[:2653], :].reset_index(drop=True)
        # training_set_0 = transactions_0.iloc[indices[2653:], :].reset_index(drop=True)
        #
        # indices = list(range(0, len(transactions_1)))
        # random.shuffle(indices)
        # test_set_1 = transactions_1.iloc[indices[:145], :].reset_index(drop=True)
        # training_set_1 = transactions_1.iloc[indices[145:], :].reset_index(drop=True)
        #
        # training_set = pd.concat([training_set_0, training_set_1], axis=0)
        # test_set = pd.concat([test_set_0, test_set_1], axis=0)
        X_train, y_train = utilities.x_y_split(training_set, 'class')
        X_test, y_test = utilities.x_y_split(test_set, 'class')

        clf = MLPClassifier(learning_rate_init=0.01, random_state=1, activation='logistic',
                            max_iter=400, hidden_layer_sizes=(256, 128, 64), )
        # verbose = True
        clf.fit(X_train, y_train)
        print("\n")
        print(clf)
        print(clf.get_params())
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f1_sum += f1
        precision = precision_score(y_test, y_pred)
        precision_sum += precision
        recall = recall_score(y_test, y_pred)
        recall_sum += recall
        probs = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, probs[:, 1])
        auc_sum += roc_auc
        print('f1 (on testset): %.4f' % f1)
        print('roc auc (on testset): %.4f' % roc_auc)

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
             str(len(y_test))],
        ]
        print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
        print(classification_report(y_test, y_pred))
        # print('\033[1m' + 'Accuracy: ' + '\033[0m' + str(accuracy))
        print(f"# of iterations: {clf.n_iter_}")
        print(f"Loss: {round(clf.loss_, 3)}")
        print(f"# of coefs: {len(clf.coefs_)}")
        print(f"Name of Output Layer Activation Function: {clf.out_activation_}")

        break

    print("\n\nAvg")
    print(f"Roc auc (class 1): {auc_sum / 10}")
    print(f"f1: {f1_sum / 10}")
    print(f"Precision: {precision_sum / 10}")
    print(f"Recall: {recall_sum / 10}")

    time_sec = time.time() - start_time
    time_min = time_sec / 60
    print("\nProcessing time of %s(): %.2f seconds (%.2f minutes)."
          % ("whole code", time.time() - start_time, time_min))
