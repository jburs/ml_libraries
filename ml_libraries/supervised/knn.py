import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

a = [1,3,3,5,2,1,9,7,8,6,9,10]
b = [2,3,3,2,1,1,8,9,9,7,8,10]
group = [1,1,1,1,1,1,2,2,2,2,2,2]

c = [2,5,7,9,6,5,6,5]
d = [2,9,2,10,6,5,5,6]

df = pd.DataFrame(list(zip(a,b,group)), columns = ['a', 'b', 'group'])
df_classification = pd.DataFrame(list(zip(c,d)), columns = ['a', 'b'])


def knn_classification(data, data_classifying, kn=5, kn_weight=True):
    ### classification column must be final column in data, and not exist in data_classifying ###
    ### knn classification algorithm with pandas data frame ###

    # Create empty datafrom for classified data
    df_classified = pd.DataFrame(columns = data.columns)

    # Loop throuph data_classifying and use knn classification for each point
    for point_index, point_row in data_classifying.iterrows():
        distances = []
        classifications = []
        weight_distance = []

        # find euclidiean distance between point and data
        for data_index, data_row in data.iterrows():
            dist = np.linalg.norm(data_row[:-1] - point_row)
            distances.append([dist, data_index])
        

        # Sort by distance (shortest to largest)
        distances =sorted(distances)
        #print('\n')
        #print(distances)

        # Classify based on kn closest points
        for k in range(kn):
            k_index = distances[k][1]                       # get data index value from k closest points
            k_classification = data.iloc[k_index][-1]         # Get classification of closest points (classification is final column)
            classifications.append(k_classification)          # Store classifications
            weight_distance.append(distances[k][0])


        # Weights for classification:  inverse weights technique
        # compute the inverse of each distance, find the sum of the inverses, then divide each inverse by the sum.
        if kn_weight == True:
           
            for i in range(len(weight_distance)):   # find 1/distance
                weight_distance[i] = 1/weight_distance[i]

            sum_weights = sum(weight_distance)      # get sum of 1/distance

            for i in range(len(weight_distance)):   # divide inverse by sum_weights
                weight_distance[i] = weight_distance[i]/sum_weights


            # Loop through unique classification values and sum weight_distance
            # Largerst value has the highest weight, and is the classification
            largest_val = 0
            for val in (set(classifications)): #loop through unique classification values (val)
                weight_dist_sum = 0
                for i in range(len(classifications)):   # sum weighted distances of knn for each classification val
                    if classifications[i] == val:
                        weight_dist_sum += weight_distance[i]
                if weight_dist_sum > largest_val:
                    largest_val = weight_dist_sum
                    classification = val


        # no weight values for classification: kn_weight == False
        else:
            classification = mode(classifications)

        # build df_classified, point row and append 'group' column with classification 
        new_row = point_row
        new_row[data.columns[-1]] = classification
        df_classified = df_classified.append(new_row)

        # Repeat for each point in data_classifying

    #print('df_classified')
    #print(df_classified.head())

    return(df_classified)



df_classified = knn_classification(df, df_classification, kn_weight=True)


plt.scatter(df['a'],df['b'], c=df['group'], marker='x')
plt.scatter(df_classified['a'],df_classified['b'], c=df_classified['group'], marker='*')
plt.show()
