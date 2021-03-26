#!/usr/bin/python3
# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
import training_data as train

DATASET = np.array([[np.array([2.7810836, 2.550537003]), 0],
                    [np.array([1.465489372, 2.362125076]), 0],
                    [np.array([3.396561688, 4.400293529]), 0],
                    [np.array([1.38807019, 1.850220317]), 0],
                    [np.array([3.06407232, 3.005305973]), 0],
                    [np.array([7.627531214, 2.759262235]), 1],
                    [np.array([5.332441248, 2.088626775]), 1],
                    [np.array([6.922596716, 1.77106367]), 1],
                    [np.array([8.675418651, -0.242068655]), 1],
                    [np.array([7.673756466, 3.508563011]), 1]])


def neuron(input_data: np.array, w: np.array, bias: float) -> float:
    """
    A single neuron, calculating w1 * x + w2 * y + w3

    :param input_data: np.array
    :param w: np.array
    :param bias: float
    :return: float
    """
    p = np.sum(input_data * w) + bias
    return 1 if p >= 0 else 0


def random_training(dataset_length: int):
    """
    Train the neural network with randomly chosen weights and bias

    :return: Loss function and weights
    """
    loss_function = np.array([])
    w1 = np.array([])  # Initialize arrays
    dataset = train.generate_dataset(dataset_length)

    for i in range(1000):
        errors = np.array([])
        weights = np.array([np.random.uniform(0, 2), np.random.uniform(0, 2)])  # Choose random weights
        # bias = np.random.uniform(-10, 10)     # Choose random bias
        bias = 0

        for d in dataset:
            input_data = np.array([train.vowel_rate(d[0])])
            res = neuron(input_data=input_data, w=weights,
                         bias=bias)  # Calculate neuron return value with chosen weights and bias
            r2 = (1 - res) ** 2 if d[1] == 1 else res ** 2  # Calculate error
            errors = np.append(errors, r2)

        loss_function = np.append(loss_function, sum(errors) / len(errors))  # Calculate loss with these values
        w1 = np.append(w1, weights[0])

    return loss_function, w1


def gradient_descent(dataset: np.array, l_rate: float = 0.1) -> tuple:
    """
    Gradient descent algorithm to train perceptron

    :param dataset:
    :param l_rate: float
    :return: tuple
    """

    # Initializing variables
    weights = np.zeros(2)
    tested_weights = [[], []]
    bias = 0
    loss_function = np.array([])

    for i in range(10):
        sum_error = 0.0

        for d in dataset:
            res = neuron(d[0], weights, bias)
            error = d[1] - res
            sum_error += error ** 2
            bias = bias + l_rate * error
            for j in range(len(weights)):
                weights[j] += l_rate * error * d[0][j]

        print(f'>>> Epoch {i}, error = {sum_error / len(dataset)}')
        for k in range(len(weights)):
            tested_weights[k].append(weights[k])
        loss_function = np.append(loss_function, sum_error / len(dataset))
    return weights, bias, tested_weights, loss_function


if __name__ == "__main__":
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(111)

    w, b, w_step, loss = gradient_descent(DATASET)
    ax1.plot(loss, np.array(w_step[0]), 'bo')
    ax1.plot(loss, np.array(w_step[1]), 'ro')
    print(w)

    plt.show()
