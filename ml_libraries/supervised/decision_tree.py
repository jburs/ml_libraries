# binary classifier decision tree

import pandas as pd
import numpy as np
import os

file_path = ('../../resources/titanic_data/train.csv')
df = pd.read_csv(os.path.dirname(__file__) + file_path)
#only keep fare, age, and survived
df = df[['Fare', 'Age', 'Survived']].copy()
df = df.dropna()

def gini_impurity(data):
    """calculate gini index for given dataset gini = 1 - sum(pi**2)"""

    prob = []
    n = len(data.index) # total data points
    classes = sorted(data.iloc[:,-1].unique()) # get classes (avoids div by 0 problem later on)

    # iterate over classes and find probabilities for each class
    for c in classes:
        prob.append(len(data[data.iloc[:,-1] == c]) / n)
    
    gini = 1 - (sum([p**2 for p in prob]))
    
    return gini



def get_split(data):
    """ test each row/column value for the best split """
    """ gets the best feature, and best value """

    best_feature = None
    best_value = 0.0
    columns = data.columns
    gini_base = gini_impurity(data)
    print('\ngini base: ', gini_base)
    n_rows = len(data.index)                # total number of rows of data before split

    # Fininding which split yields the best gini gain
    max_gain = 0

    for i in range(len(columns)-1):     # -1 b.c. class is final column
        xs = data[columns[i]].unique()  # get values to test
        for x in xs:                    # test values
            # split dataset
            df_left = data[data[columns[i]] < x]
            df_right = data[data[columns[i]] >= x]

            # get gini impurities
            gini_left = gini_impurity(df_left)
            gini_right = gini_impurity(df_right)
            

            # Calculated weighted gini impurity
            w_left = len(df_left.index) / n_rows
            w_right = len(df_right.index) / n_rows

            w_gini = gini_left * w_left + gini_right * w_right
            

            # Calculate gini gain (we want to minimize w_gini for the smallest impurity)
            gini_gain = gini_base - w_gini

            # check if this is the best split so far, store values, update max_gini
            if gini_gain > max_gain:
                best_feature = columns[i]
                best_value = x
                max_gain = gini_gain
                print(best_feature, best_value, gini_gain)

    return best_feature, best_value



print(df.head(5))
def decision_tree(data, max_depth=3, min_samples=25):
    
    depth = 0
    n = len(data.index)

    # if there is GINI to be gained, we split further
    if (depth < max_depth) and n >= min_samples:

        # Get the best split
        best_feature, best_value = get_split(data)


    return 




decision_tree(df)