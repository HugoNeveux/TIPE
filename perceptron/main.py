#!/usr/bin/python3
# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
import training_data as train


def neuron(input_data: np.array, w: np.array, bias: float) -> float:
    """
    A single neuron, calculating w1 * x + w2 * y + w3

    :param input_data: np.array
    :param w: np.array
    :param bias: float
    :return: float
    """
    p = np.sum(input_data * w) + bias
    return p


def random_training(dataset_length: int):
    """
    Train the neural network with randomly chosen weights and bias

    :return: Loss function and weights
    """
    loss_function = np.array([])
    w1 = np.array([])     # Initialize arrays
    dataset = train.generate_dataset(dataset_length)

    for i in range(1000):
        errors = np.array([])
        weights = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)])    # Choose random weights
        # bias = np.random.uniform(-10, 10)     # Choose random bias
        bias = 0

        for d in dataset:
            input_data = np.array([train.vowel_rate(d[0])])
            res = neuron(input_data=input_data, w=weights, bias=bias)   # Calculate neuron return value with chosen weights and bias
            error = (1 - res) ** 2 if d[1] == 1 else res ** 2    # Calculate error
            errors = np.append(errors, error)

        loss_function = np.append(loss_function, sum(errors) / len(errors))     # Calculate loss with these values
        w1 = np.append(w1, weights[0])

    return loss_function, w1


if __name__ == "__main__":
    fig = plt.figure(tight_layout=True)

    ax1 = fig.add_subplot(111)

    loss, we1 = random_training(200)
    ax1.plot(we1, loss, 'bo', label="Loss function")     # Plot results

    ax1.legend()
    plt.show()
