# Other models benchmark for comparison

import numpy as np
import sklearn as sk
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from koi_dataset import load_koi_dataset

# Set the seed
seed = 42
np.random.seed(seed)

# Load the dataset
x_train, x_test, y_train, y_test = load_koi_dataset()
x_data = np.concatenate((x_train, x_test))
y_data = np.concatenate((y_train, y_test))

scores = [ 'precision', 'recall', 'f1' ]

models = {
    'knn': KNeighborsClassifier,
    'svc': SVC,
    'net': MLPClassifier
}

cv = sk.model_selection.StratifiedKFold(5)

def evaluate_model(model):
    results = {}
    estimator = model()
    for score in scores:
        results[score] = sk.model_selection.cross_val_score(
            estimator, x_data, y_data, cv=cv, scoring=score
        ).mean()
    return results

for k in models:
    values = evaluate_model(models[k])
    print(k + ': ' + str(values))
