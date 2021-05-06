import numpy as np

# Sigmoid activation function
def sigmoid(z):
    soln = 1/(1+np.exp(-z))
    return(soln)

# Sigmoin derivative
def deriv_sigmoid(z):
    soln = sigmoid(z)*(1-sigmoid(z))
    return(soln)

# Tanh activation function
def tanh(z):
    soln = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return(soln)

# tanh derivative
def deriv_tanh(z):
    soln = 1/((np.cosh(z))**2)
    return(soln)

# Relu activation function 
def ReLU(z):
    soln = max(0, z)
    return(soln)

# Relu derivative
def deriv_ReLU(z):
    if z > 0:
        return(1)
    else:
        return(0)
