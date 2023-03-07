import csv
import time
import numpy as np
import pandas as pd
from daehan_mlutil import utilities
from mlxtend.preprocessing import TransactionEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from tabulate import tabulate
import random


if __name__ == '__main__':
    start_time = time.time()
    with open('../data/dataset.csv', 'r') as file:
        data_set = [list(filter(None, row)) for row in csv.reader(file)]
    # seeds [0, 10, 35, 42, 123, 456, 789, 101112, 131415, 161718]
    auc_sum = 0
    f1_sum = 0
    for seed in range(10):
        print(f"\n\nseed: {seed}")
        random.seed(seed)
        te = TransactionEncoder()
        te_ary = te.fit_transform(data_set)
        m_transactions = pd.DataFrame(te_ary, columns=te.columns_)

        transactions_0 = pd.DataFrame(
            m_transactions[m_transactions['0']].reset_index(drop=True).drop(['1', '0'], axis=1))
        transactions_1 = pd.DataFrame(
            m_transactions[m_transactions['1']].reset_index(drop=True).drop(['1', '0'], axis=1))

        indices = list(range(0, len(transactions_0)))
        random.shuffle(indices)
        transactions_te_0 = transactions_0.iloc[indices[:417], :].reset_index(drop=True)
        transactions_tr_0 = transactions_0.iloc[indices[417:], :].reset_index(drop=True)

        indices = list(range(0, len(transactions_1)))
        random.shuffle(indices)
        transactions_te_1 = transactions_1.iloc[indices[:43], :].reset_index(drop=True)
        transactions_tr_1 = transactions_1.iloc[indices[43:], :].reset_index(drop=True)

        tr_0_ary = transactions_tr_0.values.astype('int')
        tr_1_ary = transactions_tr_1.values.astype('int')
        X_train = np.concatenate((tr_0_ary, tr_1_ary), axis=0)
        y_train = np.concatenate((np.zeros(tr_0_ary.shape[0]), np.ones(tr_1_ary.shape[0])), axis=0)

        te_0_ary = transactions_te_0.values.astype('int')
        te_1_ary = transactions_te_1.values.astype('int')
        X_test = np.concatenate((te_0_ary, te_1_ary), axis=0)
        y_test = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)

        clf = MLPClassifier(learning_rate_init=0.01, random_state=1, activation='relu', max_iter=200,
                            hidden_layer_sizes=(64, 16), )
        # verbose = True
        clf.fit(X_train, y_train)
        print("\n")
        print(clf)
        print(clf.get_params())

        pred_y = clf.predict(X_test)
        f1 = f1_score(y_test, pred_y)
        f1_sum += f1
        print('Test f1 (on testset): %.3f' % f1)
        probs = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, probs[:, 1])
        auc_sum += roc_auc
        print('roc auc (on testset): %.3f' % roc_auc)

        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(pred_y.shape[0]):
            if y_test[i] == 1 and pred_y[i] == 1:
                TP = TP + 1
            elif y_test[i] == 1 and pred_y[i] == 0:
                FN = FN + 1
            elif y_test[i] == 0 and pred_y[i] == 1:
                FP = FP + 1
            else:
                TN = TN + 1

        print('pre:', TP / (TP + FP), 'rec:', TP / (TP + FN))
        print('f1:', (2 * TP) / (2 * TP + FP + FN))

        confusion_matrix = [
            ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
            ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
             str(TP + FN) + "\n(Total actual c= '1')"],
            ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
             str(FP + TN) + "\n(Total actual c= '0')"],
            ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
             str(len(X_test))],
        ]
        print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
        print(classification_report(y_test, pred_y))
        # print('\033[1m' + 'Accuracy: ' + '\033[0m' + str(accuracy))
        print(f"# of iterations: {clf.n_iter_}")
        print(f"Loss: {round(clf.loss_, 3)}")
        print(f"# of coefs: {len(clf.coefs_)}")
        print(f"Name of Output Layer Activation Function: {clf.out_activation_}")

    print("\n\nAvg")
    print(f"Roc auc (class 1): {auc_sum / 10}")
    print(f"f1: {f1_sum / 10}")
    time_sec = time.time() - start_time
    time_min = time_sec / 60
    print("\nProcessing time of %s(): %.2f seconds (%.2f minutes)."
          % ("whole code", time.time() - start_time, time_min))