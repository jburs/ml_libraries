import numpy as np
import pandas as pd


X = np.array([1.0, 5.4, 9.1, 2.7, 5.2, 9.5, 1.6, 8.9, 3.0, 5.3, 11.1])

df = pd.DataFrame(np.random.randint(0, 10, size=(5,4)), columns=list('ABCD'))
print(df.head())


# Rescaling Min-max normalization for numpy array 
def np_arr_rescaling(data, range_low=0.0, range_high=1.0):
    """ for numpy array: rescales data between range_low and range_high"""
    
    rescaled_data = np.array([])
    min_value = np.amin(data)
    max_value = np.amax(data)

    for value in data:
        value_rescaled = ((value-min_value)*(range_high-range_low)) / (max_value-min_value)
        rescaled_data=np.append(rescaled_data, value_rescaled)

    return(rescaled_data)

# Rescaling min-max normalization for pandas dataframes by row
def rescaling(data, range_low=0.0, range_high=1.0):
    """ rescales pandas data frame between range_low and range_high """

    for (column_name, column_data) in data.iteritems():
        min_value = column_data.min()
        max_value = column_data.max()

        #index counter for loc 
        index = 0
        for value in column_data:
            value_rescaled = ((value-min_value)*(range_high-range_low)) / (max_value-min_value)
            data.loc[index,column_name] = value_rescaled
            index += 1

    return(data)



# Standardization (Z-score Normalization) for np array 
def np_arr_standardizaion(data):
    """ numpy array feature standardization (Z-score Normalization) """

    rescaled_data = np.array([])
    N = data.size
    mean = np.mean(data)
    standard_deviation = np.sqrt((sum((data-mean)**2))/N)

    for value in data:
        value_rescaled = (value-mean)/standard_deviation
        rescaled_data=np.append(rescaled_data, value_rescaled)

    return(rescaled_data)
    

# Standardization (Z-score Normalization) for pandas data frame
def standardizaion(data):
    """ pandas df feature standardization (Z-score Normalization) (-ve and +ve numbers centered at 0)"""

    N = data.shape[0]  # Returns number of columns, shape[0] is rows

    for (column_name, column_data) in data.iteritems():
        mean = column_data.mean()
        standard_deviation = np.sqrt((sum((column_data-mean)**2))/N)

        #index counter for loc 
        index = 0
        for value in column_data:
            value_rescaled = (value-mean)/standard_deviation
            data.loc[index,column_name] = value_rescaled
            index += 1

    return(data)

#print(X, "\n")

#print(rescaling(df), "\n")
print(standardizaion(df), "\n")

