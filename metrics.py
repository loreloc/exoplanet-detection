import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Print some evaluation metrics
def show_metrics(rfc, x_test, y_test):
	# Get the shape of the test set
	(num_samples, num_features) = x_test.shape

	# Calculate the predictions the test set
	y_pred = rfc.predict(x_test)

	# Compute precision, recall and F1 metrics
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	# Print precision, recall and f1 metrics
	print('Precision: ' + str(round(precision, 3)))
	print('Recall: ' + str(round(recall, 3)))
	print('F1: ' + str(round(f1, 3)))

	# Compute and print the confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	print('Confusion Matrix:')
	print(cm)

	# Compute and print the first five most important features
	importances = np.round(rfc.feature_importances_, 3)
	indices = np.argsort(importances)[::-1][:5]
	print('Features Importances:')
	print(list(zip(indices, importances[indices])))