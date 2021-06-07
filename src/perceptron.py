import numpy as np


class Layer:
    """Base layer class"""
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        
    def forward(self, data: np.matrix, weights: np.matrix):
        pass
    
class Hidden:
    """Hidden layer"""
    def __init__(self, n_neurons: int, activation):
        self.n_neurons = n_neurons
        self.activation = activation
        self.output_history = []
        self.errors = []
        
    def forward(self, data: np.matrix, weights: np.matrix, bias: float):
        output = self.activation(data * weights + bias)
        self.output_history.append(output)
        return output
    
    def backward(self, error: np.matrix, weights: np.matrix):
        return weights * error
    
def ReLU(x):
    return 0 if x < 0 else x
