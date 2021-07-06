# ml_libraries
machine learning library to be used with pandas. Includes neural network, gradient descent, k nearest neighbors, and k means algorithms. 
fearure_scaling includes tools for use with the libraries. Includes data rescaling tools and standardization. 

# Packaging
https://packaging.python.org/tutorials/packaging-projects/

# Neural network
loc: neural_network / first

info: currently runs with numpy arrays, pandas support with next update. Error functions, and activation functions imported from sepatae files within neural_network directory. 

to use: import nn_first, activation functions, and error functions.
call: nn_first(X=X, Y=Y, lr=.01, acceptable_error=.005, num_hidden_nodes_1=2, num_input_nodes=1, num_output_nodes=1, max_epoch=500, accuracy_threshold=0.4):



# K means algorithm
loc:  unsupervised / k means

info: unsupervised ml algorithm, uses python and pandas dataframes to categorize data into k=n data clusters. Outputs the input dataframe with an added 'k' column for the cluster, and 'k' dataframe containg the k means.

To use: import kmeans, and call kmeans(df_data, k=3, max_epoch=25, target_kmean_shift=.005)

![kmeans_iris_plot.png](https://github.com/jburs/ml_libraries/blob/main/ml_libraries/images/kmeans_iris_plot.png)



# K nearest neighbors algorithm (knn)

info: supervised ml algorithm, uses python and pandas dataframes to catagorize unknown data based on a known set of data. Known dataset needs the group/classification as the final column. unknown dataset must have columns in the same order, without the group/classification column. Will return a new dataframe of the unknown data with a classification column added to the end. 

to use: import knn_classification, pandas, and from statistics import mode if kn_weight=False.
knn_classification(data, data_classifying, kn=5, kn_weight=True)

