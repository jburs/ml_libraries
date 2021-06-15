# lin_reg_gradient_descent.py

# linear regression with gradient descent, updating graphics per interval. 

# initialize with straight line at bet fit
# run gradient descent function to find next iteration
#    function: take in best fit params and run until finished
# store 

# y = h0 + h1*x + h2*x^2 + ...  hn*x^n

# Linear gradient descent
def gradient_descent_lin(xs, ys, h0=0, h1=0, l_rate=.001, max_epoch = 1000, target_accuracy=.001):
    num_samples = len(xs)
    accuracy = 100
    epoch = 0
    h0_history = [h0]
    h1_history = [h1]
    print("initializing")
    while epoch < max_epoch and target_accuracy < accuracy:



        # gradient descent algorithm
        new_h0 = h0 - (l_rate/num_samples)*sum([h0 + h1*xs[i]-ys[i] for i in range(num_samples)])
        new_h1 = h1 - (l_rate/num_samples)*sum([(h0 + h1*xs[i]-ys[i])*xs[i] for i in range(num_samples)])

        # accuracy check (avg dif between prev and current y_pred
        # Error: checking on best fit chance, not fit vs points
        y_pred_prev = [h0 + h1*xi for xi in xs]
        y_pred_current = [new_h0 + new_h1*xi for xi in xs]
        accuracy = sum([abs(y_pred_prev[i]-y_pred_current[i]) for i in range(num_samples)])/num_samples

        # increment variables 
        h0 = new_h0
        h1 = new_h1
        h0_history.append(h0)
        h1_history.append(h1)
        print(f"h0: {h0},  h1: {h1},  accuracy: {accuracy}")

        # increment count
        epoch += 1

    print(f"{epoch-1} iterations")
    return(h0, h1, h0_history, h1_history)



# quadratic gradient descent
def gradient_descent_quadratic(xs, ys, h0=10, h1=1, h2=2, l_rate=.0001, max_epoch = 1000, target_accuracy=.004):
    num_samples = len(xs)
    accuracy = 100
    epoch = 0
    h0_history = [h0]
    h1_history = [h1]
    h2_history = [h2]
    print("initializing")
    while epoch < max_epoch and target_accuracy < accuracy:


        # gradient descent algorithm
        new_h0 = h0 - (l_rate/num_samples)*sum([h0 + h1*xs[i] + h2*(xs[i]**2)-ys[i] for i in range(num_samples)])
        new_h1 = h1 - (l_rate/num_samples)*sum([(h0 + h1*xs[i] + h2*(xs[i]**2)-ys[i])*xs[i] for i in range(num_samples)])
        new_h2 = h2 - (l_rate/num_samples)*sum([(h0 + h1*xs[i] + h2*(xs[i]**2)-ys[i])*(xs[i]**2) for i in range(num_samples)])

        # accuracy check (avg dif between prev and current y_pred
        y_pred_prev = [h0 + h1*xi for xi in xs]
        y_pred_current = [new_h0 + new_h1*xi for xi in xs]
        accuracy = sum([abs(y_pred_prev[i]-y_pred_current[i]) for i in range(num_samples)])/num_samples

        # increment variables 
        h0 = new_h0
        h1 = new_h1
        h2 = new_h2
        h0_history.append(h0)
        h1_history.append(h1)
        h2_history.append(h2)
        print(f"h0: {h0},  h1: {h1},  h2: {h2},  accuracy: {accuracy}")

        # increment count
        epoch += 1

    return(h0, h1, h2, h0_history, h1_history, h2_history)

# Do this and make a 3D plot
def gradient_descent_lin_mv(xs, ys, zs, h0=0, h1=0, h2=0, l_rate=.001, max_epoch = 1000, target_accuracy=.001):
    num_samples = len(xs)
    accuracy = 100
    epoch = 0
    h0_history = [h0]
    h1_history = [h1]
    print("initializing")
    while epoch < max_epoch and target_accuracy < accuracy:



        # gradient descent algorithm
        new_h0 = h0 - (l_rate/num_samples)*sum([h0 + h1*xs[i]-ys[i] for i in range(num_samples)])
        new_h1 = h1 - (l_rate/num_samples)*sum([(h0 + h1*xs[i]-ys[i])*xs[i] for i in range(num_samples)])

        # accuracy check (avg dif between prev and current y_pred
        # Error: checking on best fit chance, not fit vs points
        y_pred_prev = [h0 + h1*xi for xi in xs]
        y_pred_current = [new_h0 + new_h1*xi for xi in xs]
        accuracy = sum([abs(y_pred_prev[i]-y_pred_current[i]) for i in range(num_samples)])/num_samples

        # increment variables 
        h0 = new_h0
        h1 = new_h1
        h0_history.append(h0)
        h1_history.append(h1)
        print(f"h0: {h0},  h1: {h1},  accuracy: {accuracy}")

        # increment count
        epoch += 1

    print(f"{epoch-1} iterations")
    return(h0, h1, h0_history, h1_history)


def linear_plot(h0, h1, h0s, h1s):
    # init subplots
    fig, axs = plt.subplots(1, 3, figsize = (12, 6))

    # scatter the individual points
    axs[0].scatter(x, y, marker='x', c='red')

    # plot the best fit line with epoch label
    for k in range(len(h0s)):
        y_fit = [h0s[k]+h1s[k]*xi for xi in x] #linear

        # plot every 100th iteration
        if k%50 == 0:
            axs[0].plot(x, y_fit, label=f"epoch: {k}")

    # scatter labeling
    axs[0].legend()
    axs[0].set_title('linear regression w gradient descent')
    axs[0].set_ylabel("y_values")
    axs[0].set_xlabel("x_values")

    # h0 gradient descent plot
    axs[1].plot(h0s)
    axs[1].set_title('h0 gradient descent')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('h0')

    # h1 gradient descent plot
    axs[2].plot(h1s)
    axs[2].set_title('h1 gradient descent')
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('h1')

    # show figure
    plt.show()


def quadratic_plot(h0, h1, h2, h0s, h1s, h2s):
    # init subplots
    fig, axs = plt.subplots(1, 4, figsize = (14, 6))

    # scatter the individual points
    axs[0].scatter(x, y, marker='x', c='red')

    # plot the best fit line with epoch label
    for k in range(len(h0s)):
        y_fit = [h0s[k] + h1s[k]*xi + h2s[k]*xi**2 for xi in x] # quadratic

        # plot every 100th iteration
        if k%500 == 0:
            axs[0].plot(x, y_fit, label=f"epoch: {k}")

    # scatter labeling
    axs[0].legend()
    axs[0].set_title('quadratic regression w gradient descent')
    axs[0].set_ylabel("y_values")
    axs[0].set_xlabel("x_values")

    # h0 gradient descent plot
    axs[1].plot(h0s)
    axs[1].set_title('h0 gradient descent')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('h0')

    # h1 gradient descent plot
    axs[2].plot(h1s)
    axs[2].set_title('h1 gradient descent')
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('h1')

    # h2 gradient descent plot
    axs[3].plot(h1s)
    axs[3].set_title('h2 gradient descent')
    axs[3].set_xlabel('epoch')
    axs[3].set_ylabel('h2')

    # show figure
    plt.show()

import matplotlib.pyplot as plt
import random

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y = [1, 3, 2, 5, 7, 5, 9, 4, 9, 14, 10]

#x = []
#y = []
#for i in range(25):
#    x.append(10*random.random())
#    y.append(10*random.random())


# Run linear gradient descent algorithm and plotting fn
h0, h1, h0s, h1s= gradient_descent_lin(x, y)
linear_plot(h0, h1, h0s, h1s)

# Run quadratic gradient descent alorithm
h0, h1, h2, h0s, h1s, h2s= gradient_descent_quadratic(x, y)
quadratic_plot(h0, h1, h2, h0s, h1s, h2s)

