import numpy as np
import matplotlib.pyplot as plt

class Neural_Network():
  def __init__(self, inputSize):
    #parameters
    np.random.seed(1)

    self.inputSize = inputSize
    self.hiddenSize = 9
    #Create random Weights
    self.weigth_One = 2 * np.random.random((self.inputSize, self.hiddenSize)) - 1
    self.weigth_Out = 2 * np.random.random((self.hiddenSize, 1)) - 1

    self.weigth_Out_Change = []
    self.weigth_One_Change = []


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
    self.weigth_Out_Change.append(self.Out)
    return self.Out

  def train (self, train_input, train_output, time):
    for i in range(time):
      self.think(train_input)
      Err_Out = train_output - self.Out
      Delta_Out = Err_Out * self.__sigmoid_derivative(self.Out)

      Err_One = np.dot(Delta_Out, self.weigth_Out.T)
      Delta_One = Err_One* self.__sigmoid_derivative(self.One)

      #update Weight
      self.weigth_One += .1*np.dot(train_input.T, Delta_One )
      self.weigth_Out += .1*np.dot(self.One.T, Delta_Out)

      self.weigth_One_Change.append(self.Out)




NN = Neural_Network(8)

training_set_inputs = np.array([[1, 4.7, 3.2, 1.3, 0.2], [1, 6.1, 2.8, 4.7, 1.2], [1, 5.6, 3.0, 4.1, 1.3], [1, 5.8, 2.7, 5.1, 1.9], [1, 6.5, 3.2, 5.1, 2.0]], dtype=float)
training_set_outputs = np.array([[0, 1, 1, 0, 0]], dtype=float).T

training_set_inputs = np.array([[147],[150],[153],[155],[158],[160],[163],[165],[168],[170],[173],[175],[178],[180],[183]])
training_set_outputs = np.array([[49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]]).T

training_set_inputs = np.array([[9,5,16,5,16,8,9,8],[2,5,5,5,5,11,2,11],[9,7,18,7,18,15,9,15],[8,9,17,9,17,12,8,12],[8,7,8,0,0,0,6,1],[2,2,8,5,1,6,4,2],[1,5,3,6,7,0,4,6],[2,0,9,1,3,5,9,9]])
training_set_outputs = np.array([[1,1,1,1,0,0,0,0]]).T



NN.train(training_set_inputs, training_set_outputs, 10000)



print(NN.think(np.array([4,9,5,9,5,12,4,12])))


tempb = NN.weigth_One_Change
tempa = NN.weigth_Out_Change

print(tempa, tempb)

plt.plot(tempb[0])
plt.show()
