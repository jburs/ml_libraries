import numpy as np
import random
from activation_fns import *

# Simple data for learing even vs. odd. 1=odd, 0=even

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


# Simple neural net 
#       input: 1
#       hidden layer: 3  sigmoid or tanh
#       output: 1, RelU {0, 1}
def nn_first(X=X, Y=Y, lr=.1, acceptable_error=.1, num_hidden_nodes=3, input_nodes=1, output_nodes=1):

    # Initial declarations
    num_itter = 1
    error = 1.0
    data = X
    actual = Y
    input_weights = np.array([])
    hidden_weights = np.array([])
    hidden_nodes = np.array([])

    # Fill weights
    for i in range(num_hidden_nodes):
        input_weights = np.append(input_weights, random.uniform(-1, 1))
        hidden_weights = np.append(hidden_weights, random.uniform(-1, 1))



    while error > acceptable_error and num_itter < 5:
        num_itter += 1
        for point in range(len(data)):

            # Refresh empy list for storring hidden layer nodes
            hidden_nodes_new = np.array([])
            
            # Single Input X(i)

            # Hipdden layer
            for i in range(num_hidden_nodes):        # Loop num hidden nodes and calculate each node
                z = data[point]*input_weights[i]           # Sum (input_node * input_weights) going to hidden_layer_node (only 1 input_node in this case)
                node_i = sigmoid(z)                  # Sigmoid activation function to calculate hidden_layer_node           
                hidden_nodes_new = np.append(hidden_nodes_new, node_i)      # Save hidden_layer_node value 
            #print(hidden_nodes_new)

            # Output layer single node (sum hidden_node * hidden_weights, then put through activation fn)
            z = sum(hidden_nodes_new * hidden_weights)
            prediction = ReLU(z)


            # Compute error (prediction-actual) (get real fn)
            error = prediction - actual[point]
            print(prediction, "    ", actual[point], "     ", error)

            # Compute gradient wi for all weight from hidden layer to output layer then back propogation

            # Compute gradient wi for all weights from input later to hidden layer then  Back propogation

            # update neural network weights 

            # Return neural network


    return()


nn_first()