#!/usr/bin/python3
# coding: utf8

import numpy as np

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
    r = np.sum(input_data[0:2] * w) + bias
    return 1 if r > 0.0 else 0


def gradient_descent():
    res = np.array([])
    weights = np.zeros(2)
    bias = 0
    for data in DATASET:
        res = np.append(res, neuron(data, weights, bias))
    error = sum(res - EXPECTED_RESULTS) / 10
    r2 = error**2


if __name__ == "__main__":
    gradient_descent()
