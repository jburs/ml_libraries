import numpy as np
import random
from activation_fns import *
from error_fns import *

# Simple data for learing even vs. odd. 1=odd, 0=even
#  in, hidden, out 
#      /  0  \
#   0  -  0  -  0
#      \  0  /
#
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
Y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


# Simple neural net 
#       input: 1
#       hidden layer: 3  sigmoid or tanh
#       output: 1, RelU {0, 1}
def nn_first(X=X, Y=Y, lr=.01, acceptable_error=.01, num_hidden_nodes=3, input_nodes=1, output_nodes=1, error_fn=mse):

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
        predictions = np.array([])
        for point in range(len(data)):

            # Refresh empy list for storring hidden layer nodes
            hidden_nodes_new = np.array([])
            

            # Hidden layer
            for i in range(num_hidden_nodes):        # Loop num hidden nodes and calculate each node
                z = data[point]*input_weights[i]           # Sum (input_node * input_weights) going to hidden_layer_node (only 1 input_node in this case)
                node_i = sigmoid(z)                  # Sigmoid activation function to calculate hidden_layer_node           
                hidden_nodes_new = np.append(hidden_nodes_new, node_i)      # Save hidden_layer_node value 
            
            # Update hidden nodes: Replace hidden nodes with hidden nodes new
            hidden_nodes = hidden_nodes_new

            # Output layer single node (sum hidden_node * hidden_weights, then put through activation fn)
            output_in = sum(hidden_nodes * hidden_weights)

            # Calculate model prediciton
            prediction = ReLU(output_in)
            predictions = np.append(predictions, prediction)


            # Compute Error for each output neuron E = mse
            error = mse(prediction, actual[point])


            # Backward pass: gradient of error w.r.t. weights dE/dw = dE/out*dout/dnetin*dnetin/dw
            # E = error function,  out = node output (activation fn)   net_in = sum weight*hidden_node_value
            
            # Output - Hidden layer backward pass

            # Refresh new_hidden_weights
            new_hidden_weights = np.array([])

            for i in range(len(hidden_weights)):
                dE_dout = deriv_mse(prediction, actual[i])      # Deriv error w.r.t. output
                dout_dnetin = deriv_ReLU(prediction)            # Deriv output w.r.t input to node
                dnetin_dweight = hidden_nodes[i]                # Deriv input to node w.r.t. weight we're optimizing

                dE_dweight =dE_dout*dout_dnetin*dnetin_dweight  # deriv error w.r.t. weight we are optimizing

                new_weight = hidden_weights[i] - lr*dE_dweight  # Calculate new weight

                new_hidden_weights = np.append(new_hidden_weights, new_weight)  # Store new weights to replace later


            # hidden layer - input backward pass
            # dE/dw(input) = dE/douth * douth/dneth * dneth/dw
            for i in range(len(input_weights)):
                dE_douth = deriv_mse(prediction, )     # deriv error w.r.t. hidden node output (activation dn)
                douth_dneth =      # deriv hidden node out (activation) w.r.t. net input to hidden node (sum w*input node)
                dneth_dw =      # deriv net input to hidden node w.r.t. weight we're optimizing







            # update neural network weights 
            hidden_weights = new_hidden_weights

        # Return neural network

    return()


nn_first()