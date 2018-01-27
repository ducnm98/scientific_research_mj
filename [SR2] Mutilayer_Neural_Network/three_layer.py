from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        # setting the number of nodes in layer 2 and layer 3
        # more nodes --> more confidence in predictions (?)
        l2 = 5
        l3 = 4

        # assign random weights to matrices in network
        # format is (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.synaptic_weights1 = 2 * random.random((8, l2)) - 1
        self.synaptic_weights2 = 2 * random.random((l2, l3)) - 1
        self.synaptic_weights3 = 2 * random.random((l3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # train neural network, adusting synaptic weights each time
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # pass training set through our neural network
            # a2 means the activations fed to second layer
            a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
            output = self.__sigmoid(dot(a3, self.synaptic_weights3))

            # calculate 'error'
            del4 = (training_set_outputs - output) * self.__sigmoid_derivative(output)

            # find 'errors' in each layer
            del3 = dot(self.synaptic_weights3, del4.T) * (self.__sigmoid_derivative(a3).T)
            del2 = dot(self.synaptic_weights2, del3) * (self.__sigmoid_derivative(a2).T)

            # get adjustments (gradients) for each layer
            adjustment3 = dot(a3.T, del4)
            adjustment2 = dot(a2.T, del3.T)
            adjustment1 = dot(training_set_inputs.T, del2.T)

            # adjust weights accordingly
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3

    def forward_pass(self, inputs):
        # pass our inputs through our neural network
        a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        output = self.__sigmoid(dot(a3, self.synaptic_weights3))
        return output


if __name__ == "__main__":
    # initialise single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("\nRandom starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("\nRandom starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)

    # the training set.
    training_set_inputs = array([[ 4.7, 3.2, 1.3, 0.2], [ 6.1, 2.8, 4.7, 1.2], [ 5.6, 3.0, 4.1, 1.3], [ 5.8, 2.7, 5.1, 1.9]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    training_set_inputs = array([[9, 5, 16, 5, 16, 8, 9, 8], [2, 5, 5, 5, 5, 11, 2, 11], [9, 7, 18, 7, 18, 15, 9, 15],[8, 9, 17, 9, 17, 12, 8, 12], [8, 7, 8, 0, 0, 0, 6, 1], [2, 2, 8, 5, 1, 6, 4, 2], [1, 5, 3, 6, 7, 0, 4, 6],[2, 0, 9, 1, 3, 5, 9, 9]])
    training_set_outputs = array([[1, 1, 1, 1, 0, 0, 0, 0]]).T
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("\nNew synaptic weights (layer 1) after training: ")
    print(neural_network.synaptic_weights1)
    print("\nNew synaptic weights (layer 2) after training: ")
    print(neural_network.synaptic_weights2)
    print("\nNew synaptic weights (layer 3) after training: ")
    print(neural_network.synaptic_weights3)

    # test with new input
    print("\nConsidering new situation [1,0,0] -> ?")
    print(neural_network.forward_pass(array([5,2,5,5,6,0,4,1])))