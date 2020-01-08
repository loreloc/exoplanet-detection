import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection

# Load and preprocess the KOI dataset
def load_koi_dataset(nrows=None, test_size=0.2):
    # Load the dataset
    df = pd.read_csv('nasa_koi_planets.csv', comment='#', nrows=nrows)
    # Drop useless columns (row ID and KOI name)
    df = df.drop(['loc_rowid', 'kepoi_name'], axis=1)

    # Replace rows having null values
    df = df.dropna(axis=0)
    # Remove rows having KOI disposition uncertain (e.g. CANDIDATE)
    candidateIndexes = df[df.koi_disposition == 'CANDIDATE'].index
    df = df.drop(candidateIndexes, axis=0)

    # Split the dataset in target and predictors
    df_y = df['koi_disposition']
    df_x = df.drop('koi_disposition', axis=1)
    # Preprocess the disposition
    df_y = df_y.apply(lambda x: 0 if x == 'CONFIRMED' else 1)

    # Convert to numpy arrays
    x_data = df_x.to_numpy()
    y_data = df_y.to_numpy()

    # Standardize the features
    scaler = sk.preprocessing.StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # Split the dataset in train set and test set
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
        x_data, y_data, test_size=test_size, stratify=y_data
    )

    return x_train, x_test, y_train, y_test
