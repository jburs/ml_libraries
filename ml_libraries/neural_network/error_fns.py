import numpy as np

# Regression
# Mean Squared Error (squared error loss)
def mse(predicted, actual):
    try:
        error = (1/(2*len(predicted)))*sum(((actual-predicted)**2))  # Requires numpy array for linear algebra
    except ValueError:
        print("acutal and predicted are different size")
    except TypeError:
        error =  (1/2)*((actual-predicted)**2)
    return(error)

def deriv_mse(predicted, actual):
    dmse_dpred = -(actual-predicted)
    return(soln)
     



# Classification
# Cross entropy (for classificatin with two options 0 and 1)
def cross_entropy(predicted, actual):
    error = (1/len(predicted))*sum(actual*np.log(predicted)+(1-actual)*np.log(1-predicted))
    return(error)

def deriv_cross_entropy(predicted, actual):
    dce_dpred = -((actual/predicted)-((1-actual)/(1-predicted)))
    return(dce_dpred)



