import numpy as np
import sklearn as sk
import sklearn.model_selection
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import HyperBand

from rfc_worker import RFCWorker
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
n_iterations = 32
n_workers = 4

# Load the dataset
x_data, y_data = load_koi_dataset()
# Split the dataset in train set and test set
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
	x_data, y_data, test_size=0.20, stratify=y_data
)

# Start a nameserver for hyperparameters optimization
nameserver = hpns.NameServer(run_id=PROJECT_NAME, host=LOCALHOST, port=None)
nameserver.start()

# Start the workers
workers = []
for i in range(n_workers):
	w = RFCWorker(
		x_train, y_train, nameserver=LOCALHOST, run_id=PROJECT_NAME, id=i
	)
	w.run(background=True)
	workers.append(w)

# Get the hyperparameters configuration space
hp_space = workers[0].get_configspace()

# Run an HyperBand optimizer
hb = HyperBand(
	configspace=hp_space, run_id=PROJECT_NAME,
	min_budget=min_budget, max_budget=max_budget
)
result = hb.run(n_iterations=n_iterations, min_n_workers=n_workers)

# Shutdown the optimizer and the nameserver
hb.shutdown(shutdown_workers=True)
nameserver.shutdown()

# Get the best model configuration found
id2config = result.get_id2config_mapping()
incumbent = result.get_incumbent_id()
config = id2config[incumbent]['config']
print(config)

# Build and train the best model
rfc = RFCWorker.build(config, max_budget)
rfc.fit(x_train, y_train)

# Print some evaluation metrics
show_metrics(rfc, x_test, y_test)
