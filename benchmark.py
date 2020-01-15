# Other models benchmark for comparison

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from koi_dataset import load_koi_dataset

# Set the seed
seed = 23
np.random.seed(seed)

# Load the dataset
x_train, x_test, y_train, y_test = load_koi_dataset()
x_data = np.concatenate((x_train, x_test))
y_data = np.concatenate((y_train, y_test))

models = {
    'knn': KNeighborsClassifier(),
    'svc': SVC(),
    'net': MLPClassifier()
}

cv = StratifiedKFold(5)

def evaluate_model(model):
    scores = cross_validate(
        model, x_data, y_data, cv=cv,
        scoring=['precision', 'recall', 'f1'], verbose=1
    )

    return {
        'precision': scores['test_precision'].mean(),
        'recall': scores['test_recall'].mean(),
        'f1': scores['test_f1'].mean()
    }

for k in models:
    values = evaluate_model(models[k])
    print(k + ': ' + str(values))
