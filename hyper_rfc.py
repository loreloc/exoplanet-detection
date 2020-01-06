import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Hyper Random Forest Classifier
class HyperRFC:
    # The number of estimators
    EstimatorsMin = 10
    EstimatorsMax = 100
    EstimatorsStep = 10

    # The maximum features percentage for each tree
    FeaturesPercMin = 0.1
    FeaturesPercMax = 1.0
    FeaturesPercNum = 10

    # The minimum number of samples required to split an internal node
    SamplesToSplit = [2, 4, 8, 16, 32]

    # Search for the best hyperparameters given a train set
    def search(self, x_train, y_train, **kwargs):
        # Build the hyperparameters space
        hp_space = {
            'n_estimators': np.arange(
                self.EstimatorsMin, self.EstimatorsMax + self.EstimatorsStep,
                self.EstimatorsStep
            ),
            'max_features': np.linspace(
                self.FeaturesPercMin, self.FeaturesPercMax, self.FeaturesPercNum
            ),
            'min_samples_split': self.SamplesToSplit
        }

        # Instantiate the random searcher
        model = RandomForestClassifier()
        hp_search = RandomizedSearchCV(model, hp_space, **kwargs)

        # Find the best model
        hp_search.fit(x_train, y_train)
        return hp_search.best_estimator_
