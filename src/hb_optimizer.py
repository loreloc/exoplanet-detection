import hpbandster.optimizers as hpopt
import hpbandster.core.result as hpres
import hpbandster.core.nameserver as hpns

class HBOptimizer:
	# Initialize the optimizer
	def __init__(self, host, run_id, worker_class,
			eta, min_budget, max_budget, n_iterations):
		self.host = host
		self.port = None
		self.run_id = run_id
		self.worker_class = worker_class
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget
		self.n_iterations = n_iterations

	# Start the nameserver
	def start(self, port=None):
		self.nameserver = hpns.NameServer(
			host=self.host, run_id=self.run_id, port=port
		)
		self.nameserver.start()

	# Close the nameserver
	def close(self):
		self.nameserver.shutdown()

	# Run the optimizer
	def run(self, n_workers, x_train, y_train):
		# Get the hyperparameters search space
		hp_space = self.worker_class.get_configspace()

		# Initialize the hyperband optimizer
		hb = hpopt.HyperBand(
			configspace=hp_space, run_id=self.run_id,
			eta=self.eta, min_budget=self.min_budget, max_budget=self.max_budget
		)

		# Start the workers
		workers = []
		for i in range(n_workers):
			w = self.worker_class(
				x_train, y_train, nameserver=self.host, run_id=self.run_id, id=i
			)
			w.run(background=True)
			workers.append(w)

		# Run the hyperband optimizer
		result = hb.run(n_iterations=self.n_iterations, min_n_workers=n_workers)

		# Shutdown the optimizer
		hb.shutdown(shutdown_workers=True)

		# Get the best model hyperparameters configuration found
		id2config = result.get_id2config_mapping()
		incumbent = result.get_incumbent_id()
		config = id2config[incumbent]['config']

		return config
