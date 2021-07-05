# k nearest neighbours unsupervised learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ml_libraries.feature_scaling import feature_scaling
from random import randrange
import os

file_path = "../../resources/new_iris_data.csv"

df = pd.read_csv(os.path.dirname(__file__) + file_path) # windows
#df = pd.read_csv(os.path.join(os.path.dirname( __file__ ), '..', 'resources/new_iris_data.csv')) # Mac


# Pre processing for dataset
#df_scaled = StandardScaler().fit_transform(df)
df_scaled = feature_scaling.rescaling(df)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
# Transform PCA to dataframe
df_scalled_pca = pd.DataFrame(data=df_pca, columns=['principal component 1', 'principle component 2'])



def kmeans(df_data, k=3, max_epoch=25, target_kmean_shift=.005):
    ### k means uncupervised clustering algorithm with pandas data frame ###

    print("----------------\nStarting k means")

    # Initial variable declaration
    epoch = 0
    kmeans_shift = 1
    columns = df_data.columns

    # initialize k central points randomly from data points, check uniqueness
    df_k_vals = pd.DataFrame(columns = columns)     # k central points dataframe
    b_vals = []
    i=0
    while i < k:
        b = randrange(len(df_data.index))
        if b not in b_vals:     # Check b is unique
            df_k_vals = df_k_vals.append(df_data.iloc[b], ignore_index=True)
            i+=1
        b_vals.append(b)        # Store b values



    # k means algorithm start!

    while kmeans_shift > target_kmean_shift and epoch < max_epoch:
        # Calculate nearest k to each point, store sorted by nearest k, recalculate k means, repeat

        # Loop through rows in df_data
        k_vals = []

        for data_index, data_row in df_data.iterrows():
            distances =[]

            #loop through k
            for k_index, k_row in df_k_vals.iterrows():

                # euclidean dist = normal of the difference between two vectors (points)
                dist = np.linalg.norm(data_row - k_row)
                distances.append([dist, k_index])

            # sorts distances from shortest to smallest: distances=[[dist_smalest, k], [...], [dist_largest, k]]
            distances =sorted(distances)

            # Store k value in list for adding to df
            k_vals.append(distances[0][1])

        # add row 'k' to df_data
        df_data['k'] = k_vals
        epoch+=1

        # Group df_data by k value and calculate new k means
        df_k_vals_new = df_data.groupby(['k']).mean()

        # accuracy test
        df_k_vals_diff = df_k_vals_new-df_k_vals    # subtract new - old
        df_k_vals_diff = df_k_vals_diff.abs()       # make positive vals
        kmeans_shift = (df_k_vals_diff.mean(axis=1)).mean()      # get mean of mean shift of each k

        # Set df_k_vals to the new k means
        df_k_vals = df_k_vals_new

        if kmeans_shift < target_kmean_shift:
            print("\nk means converged\n")
            return(df_data, df_k_vals)

        # Drop 'k' column in df_data
        df_data.drop(labels='k', axis=1, inplace=True)


    return(print("k means did not converge"))



df_iris_pca, df_k_vals = kmeans(df_scalled_pca)


fig, ax = plt.subplots()

ax.scatter(df_iris_pca["principal component 1"], df_iris_pca["principle component 2"], c=df_iris_pca["k"])
ax.set_title("Iris Classification")
plt.show()