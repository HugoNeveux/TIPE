#!/usr/bin/python3
# coding: utf8

import numpy as np
import matplotlib.pyplot as plt

DATASET = np.array([[2.7810836, 2.550537003],
                    [1.465489372, 2.362125076],
                    [3.396561688, 4.400293529],
                    [1.38807019, 1.850220317],
                    [3.06407232, 3.005305973],
                    [7.627531214, 2.759262235],
                    [5.332441248, 2.088626775],
                    [6.922596716, 1.77106367],
                    [8.675418651, -0.242068655],
                    [7.673756466, 3.508563011]])
EXPECTED_RESULTS = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
WEIGHTS = np.array([-0.1, 0.20653640140000007])
BIAS = -0.23418117710000003


def neuron(input_data: np.array, w: np.array, bias: float):
    """
    A single neuron, calculating w1 * x + w2 * y + w3

    :param input_data: np.array
    :param w: np.array
    :param bias: float
    :return: float
    """
    p = np.sum(input_data * w) + bias
    return p


def random_training():
    """
    Train the neural network with randomly chosen weights and bias
    :return: Loss function and weights
    """
    loss_function = np.array([])
    w1, w2 = np.array([]), np.array([])     # Initialize arrays

    for i in range(100):
        errors = np.array([])
        weights = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])    # Choose random weights
        bias = np.random.uniform(-1, 1)     # Choose random bias

        for j in range(len(DATASET)):
            res = neuron(DATASET[j], weights, bias)     # Calculate neuron return value with chosen weights and bias
            error = (EXPECTED_RESULTS[j] - res) ** 2    # Calculate error
            errors = np.append(errors, error)

        loss_function = np.append(loss_function, sum(errors) / len(errors))     # Calculate loss with these values
        w1 = np.append(w1, weights[0])
        w2 = np.append(w2, weights[1])

    return loss_function, w1, w2


if __name__ == "__main__":
    fig = plt.figure(tight_layout=True)

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    loss, we1, we2 = random_training()
    ax1.plot(we1, loss, 'bo', label="Weight 1")     # Plot results
    ax2.plot(we2, loss, 'ro', label="Weight 2")

    ax1.legend()
    ax2.legend()
    plt.show()
