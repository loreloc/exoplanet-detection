import statistics
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from koi_dataset import load_koi_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

x, y = load_koi_dataset()
(num_samples, num_features) = x.shape

scaler = sk.preprocessing.StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
    x, y, test_size=0.20, stratify=y
)

scores = [
    sk.metrics.precision_score,
    sk.metrics.recall_score,
    sk.metrics.f1_score
]

num_tests = 100

def evaluate_model(model):
    results = []
    for score in scores:
        s = 0.0
        for _ in range(num_tests):
            instance = model()
            instance.fit(x_train, y_train)
            y_pred = instance.predict(x_test)
            s += score(y_test, y_pred)
        s /= num_tests
        results.append(s)
    return results

print("knn " + str(evaluate_model(KNeighborsClassifier)))
print("svc " + str(evaluate_model(SVC)))
print("nnw " + str(evaluate_model(MLPClassifier)))
