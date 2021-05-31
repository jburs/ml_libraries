# k nearest neighbours unsupervised learning

import numpy as np
import matplotlib.pyplot as plt
from random import randrange


X = [2, 3, 1, 4, 3, 9, 9, 7, 8, 9, 15, 17, 14, 18, 16]
y = [4, 1, 2, 1, 3, 9, 6, 7, 8, 7, 18, 15, 17, 15, 19]

data = np.array([[1,2], [4,3], [3,3], [2,4], [9,6], [8,8], [7,9], [9,7], [17,19], [16,15], [17,18], [19,16]])


def knn(data, k=3, epoch_max=2, target_accuracy = .05):
    ### k nearest neighbors unsupervised machine learning algorithm ###
    
    #initial variable declaration
    accuracy = 1.0
    epoch = 1

    # initialize k central points randomly from data points, check uniqueness
    unique = False
    while unique == False:
        k_vals = np.array([])

        # Collect k random starting points
        for i in range(k):
            b = np.array(data[randrange(len(data))])
            k_vals = np.vstack([k_vals, b]) if k_vals.size else b
 
        #Uniqueness test in k_vals
        #if len(k_vals) == len(set(k_vals)):
         #   unique = True

        print(k_vals)
        unique = True

    # repeat until accuracy < target accuracy

    while accuracy > target_accuracy and epoch < epoch_max:
        
        # calculate nearest k to each point to group into k groups
        for point in data:
            distances = []
            for i in range(k):
                # euclidean dist = normal of the difference between two vectors (points)
                dist = np.linalg.norm(point - k_vals[i])

        epoch += 1


    # find the center of each k group = new k


    # check difference between old and new k against accuracy limit


knn(data)

#plt.scatter(X, y)
#plt.show()


