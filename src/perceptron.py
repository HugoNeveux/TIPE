import numpy as np


class Layer:
    """Base layer class"""
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        
    def forward(self, data: np.array, weights: np.matrix):
        pass

class Hidden:
    """Hidden layer"""
    def __init__(self, n_neurons_prec: int, activation):
        self.n_neurons = n_neurons_prec - 1
        self.activation = activation
        self.output_history = []
        self.errors = []
        
        # Weights initialization
        self.weights = np.random.rand(n_neurons_prec, self.n_neurons)
        
    def forward(self, data: np.array, weights: np.ndarray, bias: float):
        output = self.activation(np.dot(data, weights)) + bias
        self.output_history.append(output)
        return output
    
    def backward(self, error: np.array, weights: np.ndarray):
        return np.dot(weights, error)

class Input(Layer):
    """Input layer"""
    def __init__(self, input: np.array):
        assert input.shape[1] == 1
        self.data = input
        self.n_neurons = input.shape[0]
    
    def forward(self, data: np.array):
        return self.data

def ReLU(x):
    return 0 if x < 0 else x
