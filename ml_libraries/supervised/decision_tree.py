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
            

            # Calculate gini gain (we want to minimize w_gini for the smallest impurity. Ideal split is perfect Left=c1, Right=c2)
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


def check_endnode(data, depth, max_depth=3, min_samples=15):
    """ check if current node is an end node or continue splitting"""

    # check for endnode, return a prediction
    if depth >= max_depth or len(data.index) < min_samples:
        end_node = True
        return(end_node)

    # if not endnode, get split
    else:
        end_node = False
        return(end_node)




def decision_tree(data):
    
    tree = dict()
    tree_list = []
    depth = 0
    building = True
    end_node = False
    right_dfs = []
    right_depths = []



    print('\n--------  Starting  ---------\n')
    while building == True:

        # get split, increment depth, store df_right, check end node status
        if end_node == False:
            best_feature, best_value, df_left, df_right = get_split(data)
            depth += 1 
            right_dfs.append([df_right, depth]) 
            right_depths.append(depth)
            data = df_left
            end_node = check_endnode(data, depth)      

        
        if end_node == False:
            # Store split, set next data to left, increase depth
            tree_list.append(['split', best_feature, best_value, depth])
            print('split', best_feature, best_value, depth)
            data = df_left
            


        # Check there are still splits ready
        if len(right_dfs)==0: 
            building = False
            break


        if end_node == True:
            # make prediction
            class_count = data.iloc[:,-1].value_counts()                    
            classes = sorted(data.iloc[:,-1].unique())
            prediction = class_count.index[0]
            node_accuracy = class_count[prediction]/sum(class_count[i] for i in classes)
            tree_list.append([end_node, prediction, node_accuracy, depth])       # store right end node
            print(end_node, prediction, node_accuracy, depth)

            # set next data to prev right, adjust depth, remove right df from right backlog
            
            data = right_dfs.pop()[0]
            depth = right_depths.pop()

            
        
            # Check if df_right is also an end_node
            end_node = check_endnode(data, depth)
            print(end_node)






    

    return tree_list










tree = decision_tree(df)
#print(tree)





