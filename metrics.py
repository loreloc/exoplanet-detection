import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Print some evaluation metrics
def show_metrics(rfc, x_test, y_test):
	# Get the shape of the test set
	(num_samples, num_features) = x_test.shape

	# Calculate the predictions the test set
	y_pred = rfc.predict(x_test)

	# Compute precision, recall and F1 metrics
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test, y_pred, average='weighted'
	)
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
