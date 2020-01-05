import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection
from koi_dataset import load_koi_dataset
from hyper_rfc import HyperRFC

import seaborn as sb
import matplotlib.pyplot as plt

# Load the dataset
x_data, y_data = load_koi_dataset()
(num_samples, num_features) = x_data.shape

# Normalize the values
scaler = sk.preprocessing.StandardScaler()
x_data = scaler.fit_transform(x_data)

# Split the dataset in train set and test set
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
    x_data, y_data, test_size=0.20, stratify=y_data
)

# Instantiate the hyper model and search for the best model
hyper_model = HyperRFC()
forest = hyper_model.search(x_train, y_train, n_iter=200, n_jobs=4)
print(forest.get_params())

# Evaluate the best model found
y_pred = forest.predict(x_test)
cm = sk.metrics.confusion_matrix(y_test, y_pred)
precision = sk.metrics.precision_score(y_test, y_pred)
recall = sk.metrics.recall_score(y_test, y_pred)
f1 = sk.metrics.f1_score(y_test, y_pred)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))

# Plot the features importances
plt.subplot(211)
plt.title('Features Importances')
importances = forest.feature_importances_
std_importances = np.std(
    [tree.feature_importances_ for tree in forest.estimators_], axis=0
)

indices = np.argsort(importances)[::-1]
plt.bar(
    range(num_features), importances[indices],
    color='lightblue', yerr=std_importances[indices], align='center'
)
plt.xticks(range(num_features), indices)
plt.xlim([-1, num_features])
plt.xlabel('Feature')
plt.ylabel('Importance')

# Plot the confusion matrix
plt.subplot(212)
plt.title('Confusion Matrix')
df_cm = pd.DataFrame(
    cm, index=['Positive', 'Negative'], columns=['Positive', 'Negative']
)
ax = sb.heatmap(
    df_cm, annot=True,
    linewidths=2, fmt='d', cmap="Blues", cbar=False, square=True
)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
plt.ylabel('Predicted')
plt.xlabel('Actual')

# Show the graphs
plt.tight_layout()
plt.show()
