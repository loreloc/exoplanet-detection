import numpy as np
import pandas as pd

# Load and preprocess the KOI dataset
def load_koi_dataset(nrows=None):
    # Load the dataset
    df = pd.read_csv('nasa_koi_planets.csv', comment='#', nrows=nrows)
    # Drop useless columns (row ID and KOI name)
    df = df.drop(['loc_rowid', 'kepoi_name'], axis=1)

    # Replace NaN with the average of each column
    df = df.fillna(df.mean())
    # Remove rows having KOI disposition uncertain (e.g. CANDIDATE)
    candidateIndexes = df[df.koi_disposition == 'CANDIDATE'].index
    df = df.drop(candidateIndexes, axis=0)

    # Split the dataset in target and predictors
    df_y = df['koi_disposition']
    df_x = df.drop('koi_disposition', axis=1)
    # Preprocess the disposition
    df_y = df_y.apply(lambda x: 0 if x == 'CONFIRMED' else 1)

    # Apply the logarithm to all the predictors values
    for column in df_x.columns:
        df_x[column] = df_x[column].apply(lambda x: np.log10(1 + x))

    # Convert to numpy arrays
    x = df_x.to_numpy()
    y = df_y.to_numpy()
    return x, y
