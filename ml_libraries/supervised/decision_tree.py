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
    """calculate gini index for given dataset gini = 1 - sum(p(i)**2)"""

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
            

            # Calculate gini gain (we want to minimize w_gini for the smallest impurity ideal case is split is perfect Left=c1, Right=c2)
            # why not just find min w_gin instead of uding gini_gain and gini_base vaiables?
            gini_gain = gini_base - w_gini

            # check if this is the best split so far, store values, update max_gini
            if gini_gain > max_gain:
                best_feature = columns[i]
                best_value = x
                max_gain = gini_gain

    df_left = data.loc[data[best_feature] < best_value]
    df_right = data.loc[data[best_feature] >= best_value]
               

    return best_feature, best_value, df_left, df_right


def predict(split_data):
    """ makes prediction based on most common class """


    # returns the most common class and accuracy (value_counts sorts descending from largest count)
    class_count = split_data.iloc[:,-1].value_counts()
    classes = sorted(split_data.iloc[:,-1].unique())
    prediction = class_count.index[0]
    accuracy = class_count[prediction]/sum(class_count[i] for i in classes)

    return prediction, accuracy


def check_endnode(data, depth, max_depth, min_samples):
    """ check if current node is an end node """
    if depth >= max_depth or len(data.index) > min_samples:
        # make prediction
        pass

    else:
        #calculate next node and store right df 
        pass

def decision_tree(data, max_depth=3, min_samples=25):
    
    tree = dict()
    tree_list = []
    depth = 1
    building = True
    end_node = False
    right_dfs = []

    # if there is GINI to be gained, we split further
    

    #initialize root node
    best_feature, best_value, df_left, df_right = get_split(data) 
    tree_list.append([best_feature, best_value, 'root'])                 # Store split
    right_dfs.append(df_right)                      # Store df_right 


    # Next time:
    # finish check_endnode function
    # use check endnode to do tree_list append and depth tracking (maybe even splitting)

    while building == True:
        while depth <= max_depth and len(df_left.index) > min_samples:
                # Loop through left side of tree
                best_feature, best_value, df_left, df_right = get_split(df_left) 
                tree_list.append([best_feature, best_value, depth, 'left'])                 # Store split
                right_dfs.append(df_right)                      # Store df_right 
                depth += 1

        # iterate over right side
        if len(df_right)>0:
            best_feature, best_value, df_left, df_right = get_split(df_right) 
            tree_list.append([best_feature, best_value, depth, 'right'])                 # Store split
            right_dfs = right_dfs[:-1]
            depth += -1


        df_right = []
        if not df_right: 
            building = False

    





    return tree_list




tree = decision_tree(df)
print(tree)





