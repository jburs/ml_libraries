import numpy as np

# Regression
# Mean Squared Error (squared error loss)
def mse(predicted, actual):
    try:
        error = (1/len(predicted))*sum(((actual-predicted)**2))  # Requires numpy array for linear algebra
    except ValueError:
        print("acutal and predicted are different size")
    except TypeError:
        print("use numpy array for linear algebra")
    return(error)



# Classification
# Cross entropy (for classificatin with two options 0 and 1)
def cross_entropy(predicted, actual):
    error = (1/len(predicted))*sum(actual*np.log(predicted)+(1-actual)*np.log(1-predicted))
    return(error)

