import numpy as np
import sklearn as sk
import sklearn.model_selection
from rfc_worker import RFCWorker
from hb_optimizer import HBOptimizer
from metrics import show_metrics
from koi_dataset import load_koi_dataset

# Set the LOCALHOST, PROJECT_NAME constants
LOCALHOST = '127.0.0.1'
PROJECT_NAME = 'exoplanet-detection'

# Set the seed
seed = 1234
np.random.seed(seed)

# Set the parameters for hyperparameters optimization
min_budget = 8
max_budget = 216
n_iterations = 16
n_workers = 4

# Load the dataset
x_data, y_data = load_koi_dataset()
# Split the dataset in train set and test set
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
	x_data, y_data, test_size=0.20, stratify=y_data
)

# Initialize the optimizer
optimizer = HBOptimizer(
	LOCALHOST, PROJECT_NAME, worker=RFCWorker,
	min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations
)

# Start the optimizer
optimizer.start()

# Run the optimizer
config = optimizer.run(n_workers, x_train, y_train)

# Close the optimizer
optimizer.close()

# Build and train the best model
print(config)
rfc = RFCWorker.build(config, max_budget)
rfc.fit(x_train, y_train)

# Print some evaluation metrics
show_metrics(rfc, x_test, y_test)
