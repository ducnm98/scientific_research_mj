import numpy as np

class Neural_Network():
  def __init__(self, inputSize):
    #parameters
    np.random.seed(1)

    self.inputSize = inputSize
    self.hiddenSize = 3
    #Create random Weights
    self.weigth_One = 2 * np.random.random((self.inputSize, self.hiddenSize)) - 1
    self.weigth_Out = 2 * np.random.random((self.hiddenSize, 1)) - 1


  def __sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def __sigmoid_derivative(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def think(self, input_data):
    #This function also means Forward Step
    self.One = self.__sigmoid(np.dot(input_data, self.weigth_One))
    self.Out = self.__sigmoid(np.dot(self.One, self.weigth_Out))
    return self.Out

  def train (self, train_input, train_output, time):
    for i in range(time):
      self.think(train_input)
      Err_Out = train_output - self.Out
      Delta_Out = Err_Out * self.__sigmoid_derivative(self.Out)

      Err_One = np.dot(Delta_Out, self.weigth_Out.T)
      Delta_One = Err_One* self.__sigmoid_derivative(self.One)

      #update Weight
      self.weigth_One += np.dot(train_input.T, Delta_One )
      self.weigth_Out += np.dot(self.One.T, Delta_Out)


NN = Neural_Network(5)

training_set_inputs = np.array([[1, 4.7, 3.2, 1.3, 0.2], [1, 6.1, 2.8, 4.7, 1.2], [1, 5.6, 3.0, 4.1, 1.3], [1, 5.8, 2.7, 5.1, 1.9], [1, 6.5, 3.2, 5.1, 2.0]])
training_set_outputs = np.array([[0, 1, 1, 0, 0]]).T

NN.train(training_set_inputs, training_set_outputs, 10000)

print(NN.think(np.array([1, 5.8, 2.7, 3.9, 1.2])))



