import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Hyper Random Forest Classifier
class HyperRFC:
    # The minimum, maximum number of estimators and its step size
    EstimatorsMin = 10
    EstimatorsMax = 120
    EstimatorsStep = 10

    # The minimum, maximum for minimum samples split and its step size
    SampleSplitMin =  2
    SampleSplitMax = 32
    SampleSplitStep = 1

    # The minimum and maximum max features percentage and the intervals number
    FeaturesPercMin = 0.1
    FeaturesPercMax = 1.0
    FeaturesPercNum = 10

    # The split criterions
    SplitCriterions = ('gini', 'entropy')

    def search(self, x_train, y_train, cv=5, n_iter=10, n_jobs=None):
        # Build the hyperparameters space
        hp_space = {
            'n_estimators': np.arange(
                self.EstimatorsMin, self.EstimatorsMax, self.EstimatorsStep
            ),
            'min_samples_split': np.arange(
                self.SampleSplitMin, self.SampleSplitMax, self.SampleSplitStep
            ),
            'max_features': np.linspace(
                self.FeaturesPercMin, self.FeaturesPercMax, self.FeaturesPercNum
            ),
            'criterion': self.SplitCriterions,
        }

        # Instantiate the random searcher
        model = RandomForestClassifier()
        hp_search = RandomizedSearchCV(
            model, hp_space, n_iter,
            scoring='f1', cv=cv, n_jobs=n_jobs
        )

        # Find the best model
        hp_search.fit(x_train, y_train)
        return hp_search.best_estimator_
