import numpy as np
import sklearn as sk
import sklearn.model_selection
from rfc_worker import RFCWorker
from hb_optimizer import HBOptimizer
from metrics import compute_metrics
from koi_dataset import load_koi_dataset

# Set the LOCALHOST, PROJECT_NAME constants
LOCALHOST = '127.0.0.1'
PROJECT_NAME = 'exoplanet-detection'

# Set the parameters for hyperparameters optimization
eta = 3
min_budget = 8
max_budget = 216
n_iterations = 8
n_workers = 4
n_repetitions = 10

# Load the dataset
x_data, y_data = load_koi_dataset()
(n_samples, n_features) = x_data.shape

# Initialize the optimizer
optimizer = HBOptimizer(
	LOCALHOST, PROJECT_NAME, RFCWorker,
	eta, min_budget, max_budget, n_iterations
)

metrics = {
	'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
	'confusion': [[0, 0], [0, 0]], 'importances': np.zeros(n_features)
}

# Repeat multiple times the test
for _ in range(n_repetitions):
	# Split the dataset in train set and test set
	x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
		x_data, y_data, test_size=0.20, stratify=y_data
	)

	# Start the optimizer
	optimizer.start()

	# Run the optimizer
	config = optimizer.run(n_workers, x_train, y_train)

	# Build and train the best model
	rfc = RFCWorker.build(config, max_budget)
	rfc.fit(x_train, y_train)

	# Compute some evaluation metrics
	scores = compute_metrics(rfc, x_test, y_test)
	for k in metrics:
		metrics[k] = metrics[k] + scores[k]

	# Close the optimizer
	optimizer.close()

# Normalize the metrics
for k in metrics:
	metrics[k] = metrics[k] / n_repetitions

# Print the metrics
print(metrics)
