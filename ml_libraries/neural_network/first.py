import numpy as np
import random
import activation_fns as afn
import error_fns as efn
import matplotlib.pyplot as plt 

# Simple data for learing even vs. odd. 1=odd, 0=even
#  in, hidden, out 
#      /  0  \
#   0  -  0  -  0
#      \  0  /
#

# Even / odd prediction
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
Y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Divide by 2
#X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
#Y = np.array([.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])

# Simple neural net 


def nn_first(X=X, Y=Y, lr=.01, acceptable_error=.005, num_hidden_nodes_1=2, num_input_nodes=1, num_output_nodes=1, max_epoch=500, accuracy_threshold=0.4):
    print('initializing')
    print('---------------------\n')

    # Initial declarations
    epoch = 0
    loss_per_epoch = 1.0
    data = X
    actual = Y
    input_weights = np.array([])
    hidden_weights_1 = np.array([])
    hidden_nodes_1 = np.zeros(num_hidden_nodes_1)
    input_bias = np.array([])
    hidden_bias_1 = np.array([])
    loss_training = np.array([])
    accuracy_training = np.array([])

    # Initial activation fns and deriv, error fn declarations
    hidden_1_activ_fn = afn.sigmoid
    hidden_1_active_fn_deriv = afn.deriv_sigmoid
    output_activ_fn = afn.sigmoid
    output_active_fn_deriv = afn.deriv_sigmoid
    error_fn = efn.cross_entropy
    error_fn_deriv = efn.deriv_cross_entropy


    # Fill weights
    for i in range(num_hidden_nodes_1):
        input_weights = np.append(input_weights, random.uniform(-1.0, 1.0))
        hidden_weights_1 = np.append(hidden_weights_1, random.uniform(-1.0, 1.0))
    
    # Set random bias
    input_bias = np.append(input_bias, random.uniform(-1.0, 1.0))
    hidden_bias_1 = np.append(hidden_bias_1, random.uniform(-1.0, 1.0))


    # Initial loop that checks the overall error of the model, or terminates after too many iterations
    while loss_per_epoch > acceptable_error and max_epoch > epoch:
        epoch += 1
        predictions = np.array([])

        # reset loss and accuracy per epoch
        loss_per_epoch = 0.
        accuracy_per_epoch = 0.

        # test nn with each point within dataset
        for point in range(len(data)):

            # Refresh empy lists for storring hidden layer nodes and weights, net inputs to nodes, and input weights
            hidden_nodes_1_new = np.array([])
            hidden_weights_1_new = np.array([])
            net_in_hidden_1 = np.array([])
            net_in_output = np.array([])
            input_weights_new = np.array([])

            # Hidden layer: calc net in (z) -> activation fn -> appen to hidden node i 
            for i in range(num_hidden_nodes_1):                   # Loop num hidden nodes and calculate each node
                z = data[point]*input_weights[i] + input_bias   # Sum (input_node * input_weights) going to hidden_layer_node (only 1 input_node in this case)
                net_in_hidden_1 = np.append(net_in_hidden_1, z)     # store net input to each hidden node for later back propgation
                node_i = hidden_1_activ_fn(z)                    # Sigmoid activation function to calculate hidden_layer_node           
                hidden_nodes_1[i] = node_i                        # replace hidden node with new value


            # Net in Output layer, single node (sum hidden_node * hidden_weights + bias)
            output_in = sum(hidden_nodes_1 * hidden_weights_1) + hidden_bias_1
            net_in_output = np.append(net_in_output, output_in)             # Store value for use in back propogarion


            # Calculate model prediciton from net output_in
            prediction = output_activ_fn(output_in)
            predictions = np.append(predictions, prediction)

            # Compute Error and accuracy for each output neuron E = mse
            error = error_fn(prediction, actual[point])
            accuracy = (prediction - actual[point])
            if accuracy < accuracy_threshold:
                accuracy_per_epoch = accuracy_per_epoch + 1
            
                
            # Store total error for epoch, to later average for a loss per epoch value
            loss_per_epoch = loss_per_epoch + abs(error)
            
            


            # Back propogation Start!
            # Swap to delta functions?? 
            #   Output - Hidden layer backward pass:    delta_out_1 = -(target_O1-out_O1)*out_O1(1-out_O1)
            #                                           dE/dweight_1 = delta_out_1*out_h1
            #
            #   hidden layer - input backward pass:     dE/dweight_1 = (sum_O(delta_out_i * weight_hi) * out_h1(1-out_h1) * input_1)
            #                                           dE/dweight_1 = delta_h1*input_1

            # Output - Hidden layer backward pass
            # Backward pass: gradient of error w.r.t. weights dE/dw = dE/outO*doutO/dnetO*dnetO/dw
            # E = error function,  out = node output (activation fn)   net_in = sum(weight*hidden_node_value) + bias
            for i in range(len(hidden_weights_1)):
                dE_doutO = error_fn_deriv(prediction, actual[point])     # Deriv error w.r.t. output
                doutO_dnetO = output_active_fn_deriv(output_in)              # Deriv output w.r.t input to node
                dnetO_dw = hidden_nodes_1[i]                          # Deriv input to node w.r.t. weight we're optimizing

                dE_dweight = dE_doutO*doutO_dnetO*dnetO_dw          # deriv error w.r.t. weight we are optimizing

                new_weight = hidden_weights_1[i] - lr*dE_dweight      # Calculate new weight

                hidden_weights_1_new = np.append(hidden_weights_1_new, new_weight)  # Store new weights to replace later

            # hidden layer - input backward pass (net=net in to node0) (out = output of node (activation fn))
            # dE/dw(input) = dE/douth * douth/dneth * dneth/dw  (each hidden-input layer weight contributes to the output and error of multiple neutons)
            # dE/douth = dE/dnetO * dnetO/douth
            # dE/dnetO = dE/doutO * doutO/dnetO
            # Thus dE/dwi = (sum Output neurons (dE/doutO * doutO/dnetO * dnetO/douth)) * douth/dneth * dneth/dw
            for i in range(len(input_weights)):
                dE_doutO = error_fn_deriv(prediction, actual[point])             # Deriv error neuron w.r.t ouput neurons
                doutO_dnetO = output_active_fn_deriv(output_in)             # Deriv output w.r.t input to output node
                dnetO_douth = hidden_weights_1[i]                           # Deriv input to output node w.r.t. hidden node output
                douth_dneth = hidden_1_active_fn_deriv(net_in_hidden_1[i])  # Deriv hidden node output w.r.t. net input to hidden node
                dneth_dw = data[point]    # Deriv net input to hidden node w.r.t. weight we are optimizing
                
                # For singular outpout neuron, otherwise sum over outputs for (dE_doutO * doutO_dnetO * dnetO_douth)
                dE_dw = dE_doutO * doutO_dnetO * dnetO_douth * douth_dneth * dneth_dw  # Derive error w.r.t. node we're optimizing

                new_weight = input_weights[i] - lr*dE_dw

                input_weights_new = np.append(input_weights_new, new_weight)


            # update neural network weights 
            hidden_weights_1 = hidden_weights_1_new
            input_weights = input_weights_new



        # epoch finished, store loss and accuracy 
        loss_per_epoch = loss_per_epoch/len(data)
        loss_training = np.append(loss_training, loss_per_epoch)
        accuracy_per_epoch = accuracy_per_epoch/len(data)
        accuracy_training = np.append(accuracy_training, accuracy_per_epoch)

        print(loss_per_epoch)


        # print out # epoch
        if epoch%10 == 0:
            print('epoch: ', epoch)
            print(input_weights)
            print(hidden_weights_1)
            print(prediction, actual[point])
            print('-----------------------------')

        


    # option to return predictions rounded for classification
    predictions_round = np.round(predictions, 3)
        


    print('epoch: ', epoch)
    results = list(zip(predictions_round, actual))
    print(results)

    # Create history variable to pass to return


    # Return neural network
    return(predictions, loss_training, accuracy_training)





predictions, loss, accuracy = nn_first()



plt.subplot(1,2,1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.plot(loss)


plt.subplot(1,2,2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.plot(accuracy)
plt.show()
