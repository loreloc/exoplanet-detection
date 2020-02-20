import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# The Random Forest Classifier hyperparameters searching worker
class RFCWorker(Worker):
	# Initialize the worker using the train set
	def __init__(self, x_train, y_train, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.x_train = x_train
		self.y_train = y_train

	# Cross-validate a random forest hyperparameters configuration
	def compute(self, config, budget, **kwargs):
		rfc = self.build(config, budget)

		scores = cross_val_score(
			rfc, self.x_train, self.y_train,
			scoring='f1', cv=StratifiedKFold(5)
		)

		return ({
			'loss': 1.0 - scores.mean()
		})

	# Get the random forest hyperparameters configuration space
	@staticmethod
	def get_configspace():
		cs = CS.ConfigurationSpace()

		# The split criterion
		criterion = CSH.CategoricalHyperparameter(
			name='criterion', choices=['gini', 'entropy']
		)

		# The maximum percentage of features to use for each tree
		max_features = CSH.UniformFloatHyperparameter(
			name='max_features', lower=0.2, upper=0.8
		)

		# The maximum fraction of samples to use to train each tree
		max_samples = CSH.UniformFloatHyperparameter(
			name='max_samples', lower=0.25, upper=1.0
		)

		# The maximum depth
		max_depth = CSH.UniformIntegerHyperparameter(
			name='max_depth', lower=8, upper=32
		)

		# The minimum number of samples required to split an internal node
		min_samples_split = CSH.UniformIntegerHyperparameter(
			name='min_samples_split', lower=2, upper=16
		)

		# The minimum number of samples required to be at a leaf node
		min_samples_leaf = CSH.UniformIntegerHyperparameter(
			name='min_samples_leaf', lower=1, upper=8
		)

		# Add the hyperparameters to the configuration space
		cs.add_hyperparameters([
			criterion,
			max_features,
			max_samples,
			max_depth,
			min_samples_split,
			min_samples_leaf
		])

		return cs

	# Build a random forest classifier using its hyperparameters configuration
	# The budget rappresents the number of trees in the forest
	@staticmethod
	def build(config, budget):
		return RandomForestClassifier(
			n_estimators=int(budget),
			criterion=config['criterion'],
			max_features=config['max_features'],
			max_samples=config['max_samples'],
			max_depth=config['max_depth'],
			min_samples_split=config['min_samples_split'],
			min_samples_leaf=config['min_samples_leaf']
		)
