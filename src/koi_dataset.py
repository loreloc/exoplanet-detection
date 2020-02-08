import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess the KOI dataset
def load_koi_dataset(nrows=None):
    # Load the dataset
    df = pd.read_csv('nasa_koi_planets.csv', comment='#', nrows=nrows)
    # Drop useless columns (row ID and KOI name)
    df = df.drop(['loc_rowid', 'kepoi_name'], axis=1)

    # Remove samples having null values
    df = df.dropna(axis=0)

    # Remove rows having KOI disposition uncertain (e.g. CANDIDATE)
    candidateIndexes = df[df.koi_disposition == 'CANDIDATE'].index
    df = df.drop(candidateIndexes, axis=0)

    # Split the dataset in target and predictors
    df_y = df['koi_disposition']
    df_x = df.drop('koi_disposition', axis=1)
    # Preprocess the disposition
    df_y = df_y.apply(lambda x: 1 if x == 'CONFIRMED' else 0)

    # Convert to numpy arrays
    x_data = df_x.to_numpy()
    y_data = df_y.to_numpy()

    # Print some statistics
    #print(x_data.shape) # (5860, 21)
    #print(y_data.mean()) # 0.38
    #import matplotlib.pyplot as plt
    #plt.matshow(df_x.corr())
    #plt.colorbar()
    #plt.show()
    #quit()

    # Standardize the features
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    return x_data, y_data
