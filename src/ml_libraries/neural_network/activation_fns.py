import numpy as np
import matplotlib.pyplot as plt

# Hidden layers are best represented with non-linearity for deeper representations
# Sigmoid, tanh, RelU
# Output layers are generally used with Linear, sigmoid, softmax
# Multilayer Perceptron (MLP): ReLU activation function.
# Convolutional Neural Network (CNN): ReLU activation function.
# Recurrent Neural Network: Tanh and/or Sigmoid activation function.




# Sigmoid activation function (logistic fn)
# Outputs between 0, 1  succeptible to vanishing gradient problem
# scale input data to the range 0-1 (e.g. the range of the activation function) prior to training.
def sigmoid(z):
    soln = 1.0/(1.0+np.exp(-z))
    return(soln)

# Sigmoid derivative
def deriv_sigmoid(z):
    soln = sigmoid(z)*(1.0-sigmoid(z))
    return(soln)


# Tanh activation function
# Outputs -1, 1  succeptible to vanishing gradient problem
# scale input data to the range -1 to 1 (e.g. the range of the activation function) prior to training.
def tanh(z):
    soln = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return(soln)

# tanh derivative
def deriv_tanh(z):
    soln = 1.0/((np.cosh(z))**2)
    return(soln)


# Relu activation function 
# less susceptible to vanishing gradients that prevent deep models from being trained, although it can suffer from other problems like saturated or “dead” units.
# Normalize input data for 0-1 (good practice)
def ReLU(z):
    soln = np.maximum(0.0, z)
    return(soln)

# Relu derivative
def deriv_ReLU(z):
    if z > 0.0:
        return(1.0)
    else:
        return(0.0)


# Linear activation function (identity)
def linear(z):
    return(z)

def deriv_linear(z):
    return(1)


# Softmax
# outputs a vector of values that sum to 1.0 that can be interpreted as probabilities of class membership.
# Target labels used to train a model with the softmax activation function in the output layer will be vectors with 1 for the target class and 0 for all other classes.
def softmax(z):
    soln_vector = np.array([])
    for element in z:
        soln = np.exp(element) / np.exp(z).sum()
        soln_vector = np.append(soln_vector, soln)
    return(soln_vector)

# partial derivative of the i-th output w.r.t. the j-th input (input = j_vector, output = i_vector)
def deriv_softmax(soft_in_j, soft_out_i):
    soln_matrix = np.array([])
    for i in range(soft_out_i.size):            # iterate through i and j
        row = np.array([])
        for j in range(soft_in_j.size):
            if i == j:                          # Calculate delta
                delta = 1
            else:
                delta = 0

            soln = soft_out_i[i]*(delta-soft_out_i[j])          # soln Find element value
            row = np.append(row, soln)                          # Append until row is complete
        soln_matrix = np.append(soln_matrix, row)               # append row to soln_matrix
    soln_matrix = soln_matrix.reshape((soft_out_i.size, soft_in_j.size)) #reshape to ixj 2D matrix
    return(soln_matrix)




# Function plots

# define input data
#inputs = [x for x in range(-10, 10)]
#sigmoid_out = [sigmoid(x) for x in inputs]
#tanh_out = [tanh(x) for x in inputs]
#ReLU_out = [ReLU(x) for x in inputs]
#linear_out = [linear(x) for x in inputs]

# softmax is probability list and cannot be graphed
#soft_inputs = np.arange(5.0)
#softmax_out = softmax(soft_inputs)
#deriv_softmax(soft_inputs, softmax_out)



# Setup subplots


#plt.subplot(3,3,1)
#plt.title('Sigmoid')
#plt.plot(inputs, sigmoid_out)

#plt.subplot(3,3,2)
#plt.title('tanh')
#plt.plot(inputs, tanh_out)

#plt.subplot(3,3,3)
#plt.title('ReLU')
#plt.plot(inputs, ReLU_out)

#plt.subplot(3,3,4)
#plt.title('Linear')
#plt.plot(inputs, linear_out)

#plt.show()
