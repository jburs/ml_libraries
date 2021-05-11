import numpy as np
import random
from activation_fns import sigmoid, deriv_sigmoid
from error_fns import mse, deriv_mse
import matplotlib.pyplot as plt 

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
#       hidden layer: 3  sigmoid 
#       output: 1, sigmoid 
def nn_first(X=X, Y=Y, lr=.05, acceptable_error=.005, num_hidden_nodes=3, num_input_nodes=1, num_output_nodes=1, error_fn=mse, max_epoch=1000, accuracy_threshold=0.4):
    print('initializing')
    print('---------------------\n')

    # Initial declarations
    epoch = 0
    error = 1.0
    data = X
    actual = Y
    input_weights = np.array([])
    hidden_weights = np.array([])
    hidden_nodes = np.array([])
    input_bias = np.array([])
    hidden_bias = np.array([])
    loss_training = np.array([])
    accuracy_training = np.array([])

    # Fill weights
    for i in range(num_hidden_nodes):
        input_weights = np.append(input_weights, random.uniform(-1, 1))
        hidden_weights = np.append(hidden_weights, random.uniform(-1, 1))
    
    # Set random bias
    input_bias = np.append(input_bias, random.uniform(-0.2, .2))
    hidden_bias = np.append(hidden_bias, random.uniform(-0.2, 2))


    # Initial loop that checks the overall error of the model, or terminates after too many iterations
    while error > acceptable_error and max_epoch > epoch:
        epoch += 1
        predictions = np.array([])

        # reset loss and accuracy per epoch
        loss_per_epoch = 0
        accuracy_per_epoch = 0

        # test nn with each point within dataset
        for point in range(len(data)):

            # Refresh empy lists for storring hidden layer nodes and weights, net inputs to nodes, and input weights
            hidden_nodes_new = np.array([])
            hidden_weights_new = np.array([])
            net_in_hidden = np.array([])
            net_in_output = np.array([])
            input_weights_new = np.array([])

            # Hidden layer: calc net in (z) -> activation fn -> appen to hidden node i 
            for i in range(num_hidden_nodes):                   # Loop num hidden nodes and calculate each node
                z = data[point]*input_weights[i] + input_bias   # Sum (input_node * input_weights) going to hidden_layer_node (only 1 input_node in this case)
                net_in_hidden = np.append(net_in_hidden, z)     # store net input to each hidden node for later back propgation
                node_i = sigmoid(z)                             # Sigmoid activation function to calculate hidden_layer_node           
                hidden_nodes_new = np.append(hidden_nodes_new, node_i)      # Save hidden_layer_node value 
            
            # Update hidden nodes: Replace hidden nodes with hidden nodes new
            hidden_nodes = hidden_nodes_new

            # Net in Output layer, single node (sum hidden_node * hidden_weights + bias)
            output_in = sum(hidden_nodes * hidden_weights) + hidden_bias
            net_in_output = np.append(net_in_output, output_in)             # Store value for use in back propogarion


            # Calculate model prediciton from net output_in
            prediction = sigmoid(output_in)
            predictions = np.append(predictions, prediction)

            # Compute Error and accuracy for each output neuron E = mse
            error = mse(prediction, actual[point])
            accuracy = (prediction - actual[point])
            if accuracy < accuracy_threshold:
                accuracy_per_epoch = accuracy_per_epoch + 1
            
                



            # Store total error for epoch, to later average for a loss per epoch value
            loss_per_epoch = loss_per_epoch + error
            
            






            # Output - Hidden layer backward pass
            # Backward pass: gradient of error w.r.t. weights dE/dw = dE/outO*doutO/dnetO*dnetO/dw
            # E = error function,  out = node output (activation fn)   net_in = sum(weight*hidden_node_value) + bias
            for i in range(len(hidden_weights)):
                dE_doutO = deriv_mse(prediction, actual[point])     # Deriv error w.r.t. output
                doutO_dnetO = deriv_sigmoid(output_in)              # Deriv output w.r.t input to node
                dnetO_dw = hidden_nodes[i]                          # Deriv input to node w.r.t. weight we're optimizing

                dE_dweight = dE_doutO*doutO_dnetO*dnetO_dw          # deriv error w.r.t. weight we are optimizing

                new_weight = hidden_weights[i] - lr*dE_dweight      # Calculate new weight

                hidden_weights_new = np.append(hidden_weights_new, new_weight)  # Store new weights to replace later

            # hidden layer - input backward pass (net=net in to node0) (out = output of node (activation fn))
            # dE/dw(input) = dE/douth * douth/dneth * dneth/dw  (each hidden-input layer weight contributes to the output and error of multiple neutons)
            # dE/douth = dE/dnetO * dnetO/douth
            # dE/dnetO = dE/doutO * doutO/dnetO
            # Thus dE/dwi = ((dE/doutO * doutO/dnetO) * dnetO/douth) * douth/dneth * dneth/dw
            for i in range(len(input_weights)):
                dE_doutO = deriv_mse(prediction, actual[point])
                doutO_dnetO = deriv_sigmoid(output_in)
                dnetO_douth = hidden_weights[i]
                douth_dneth = deriv_sigmoid(net_in_hidden[i])
                dneth_dw = data[point]

                new_weight = dE_doutO * doutO_dnetO * dnetO_douth * douth_dneth * dneth_dw

                input_weights_new = np.append(input_weights_new, new_weight)


            # update neural network weights 
            hidden_weights = hidden_weights_new
            input_weights = input_weights_new


        # epoch finished, store loss and accuracy 
        loss_per_epoch = loss_per_epoch/len(data)
        loss_training = np.append(loss_training, loss_per_epoch)
        accuracy_per_epoch = accuracy_per_epoch/len(data)
        accuracy_training = np.append(accuracy_training, accuracy_per_epoch)

        


        # print out # epoch
        if epoch%10 == 0:
            print('epoch: ', epoch)
            print(input_weights)
            print(hidden_weights)
            print(prediction, actual[point])
            print('-----------------------------')

        



    predictions_round = np.round(predictions)
        
    print('epoch: ', epoch)
    results = list(zip(predictions_round, actual))
    print(results)

    # Create history variable to pass to return


    # Return neural network
    return(predictions, loss_training, accuracy_training)





predictions, loss, accuracy = nn_first()





plt.plot(loss)
plt.show()

plt.plot(accuracy)
plt.show()
