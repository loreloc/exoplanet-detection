import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Compute some evaluation metrics for random forests
def compute_metrics(rfc, x_test, y_test):
	# Calculate the predictions the test set
	y_pred = rfc.predict(x_test)

	# Compute precision, recall and F1 metrics
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	# Compute the confusion matrix
	confusion = confusion_matrix(y_test, y_pred)

	# Compute the most important features
	importances = rfc.feature_importances_

	return {
		'precision': precision, 'recall': recall, 'f1': f1,
		'confusion': confusion, 'importances': importances
	}
