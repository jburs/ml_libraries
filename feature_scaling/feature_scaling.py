import numpy as np


X = np.array([1.0, 5.4, 9.1, 2.7, 5.2, 9.5, 1.6, 8.9, 3.0, 5.3, 11.1])


# Rescaling Min-max normalization
def rescaling(data, range_low=0.0, range_high=1.0):
    """ rescales data between range_low and range_high"""
    
    rescaled_data = np.array([])
    min_value = np.amin(data)
    max_value = np.amax(data)

    for value in data:
        value_rescaled = ((value-min_value)*(range_high-range_low)) / (max_value-min_value)
        rescaled_data=np.append(rescaled_data, value_rescaled)

    return(rescaled_data)



# Standardization (Z-score Normalization)
def standardizaion(data):
    """ feature standardization (Z-score Normalization) """

    rescaled_data = np.array([])
    N = data.size
    mean = np.mean(data)
    standard_deviation = np.sqrt((sum((data-mean)**2))/N)

    for value in data:
        value_rescaled = (value-mean)/standard_deviation
        rescaled_data=np.append(rescaled_data, value_rescaled)

    return(rescaled_data)
    

print(X, "\n")

print(rescaling(X), "\n")
print(standardizaion(X), "\n")

