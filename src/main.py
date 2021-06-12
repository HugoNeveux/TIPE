from perceptron import *

def main():
    data = np.array([[3], [2.5], [4], [3]])
    network = [Input(data)]
    network_outputs = []
    for i in range(data.shape[0]):
        layer = Hidden(network[i - 1].n_neurons, lambda x: x)
        network.append(layer)
    print(network)
    for neuron in network:
        print(neuron.n_neurons)
        
    # Network forwarding to get output

if __name__ == "__main__":
    main()
