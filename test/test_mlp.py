from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# clf.predict_proba(X_test[:1])
# clf.predict(X_test[:5, :])

score = clf.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Score: {score}")
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
