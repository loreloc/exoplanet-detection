import numpy as np
import sklearn as sk
import sklearn.model_selection
from koi_dataset import load_koi_dataset
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
x_data, y_data = load_koi_dataset()
(n_samples, n_features) = x_data.shape

cv = StratifiedKFold(5)

scores = cross_validate(
    RandomForestClassifier(), x_data, y_data, cv=cv,
    scoring=['precision', 'recall', 'f1'], verbose=1
)

print('precision: ' + str(scores['test_precision'].mean()))
print('recall: ' + str(scores['test_recall'].mean()))
print('f1: ' + str(scores['test_f1'].mean()))
