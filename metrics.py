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

	# Compute and print precision, recall and F1 metrics
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	print('Precision: ' + str(precision))
	print('Recall: ' + str(recall))
	print('F1: ' + str(f1))

	# Compute and print the confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	print('Confusion Matrix:')
	print(cm)

	# Compute and print the first five most important features
	importances = rfc.feature_importances_
	indices = np.argsort(importances)[::-1][:5]
	print('Features Importances:')
	print(list(zip(indices, importances[indices])))
