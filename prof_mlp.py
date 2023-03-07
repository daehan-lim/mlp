import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import csv
from mlxtend.preprocessing import TransactionEncoder
from tabulate import tabulate

with open('data/dataset.csv', 'r') as file:
    data_set = [list(filter(None, row)) for row in csv.reader(file)]
te = TransactionEncoder()
te_ary = te.fit_transform(data_set)
m_transactions = pd.DataFrame(te_ary, columns=te.columns_)

transactions_0 = pd.DataFrame(
    m_transactions[m_transactions['0']].reset_index(drop=True).drop(['1', '0'], axis=1))
transactions_1 = pd.DataFrame(
    m_transactions[m_transactions['1']].reset_index(drop=True).drop(['1', '0'], axis=1))

random.seed(10)
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

tuned_parameters = [{"activation": ["relu", "logistic", "tanh"], "hidden_layer_sizes": [(64, 16), (128, 64, 16)]}]
# tuned_parameters = [{"activation": ("relu", "logistic", "tanh"), "hidden_layer_sizes": [[64,16],[128,64,16]]}]

mlp = MLPClassifier(learning_rate_init=0.01, max_iter=200, random_state=1)
clf_grid = GridSearchCV(mlp, tuned_parameters, cv=10, scoring='f1', verbose=True)

clf_grid.fit(X_train, y_train)

print(clf_grid.best_params_)

print("")
print(clf_grid)
print(f"cv={clf_grid.cv}, max_iter = {mlp.max_iter}")
print(f"\nBest params: {clf_grid.best_params_}")

print(f"Best f1: {clf_grid.best_score_}")

pred_y = clf_grid.predict(X_test)
print('Test f1 (on testset): %.3f' % f1_score(y_test, pred_y))
probs = clf_grid.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, probs[:, 1])
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
