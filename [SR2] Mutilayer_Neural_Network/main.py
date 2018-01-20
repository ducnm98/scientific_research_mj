import numpy as np

#Trường hợp có 2 Lớp hiddenPlayer
#Gọi lớp HiddenPlayer 1 là One
#Gọi lớp HiddenPlayer 2 là Two
#Gọi Output là Out
class mutiplayer_neural_network:
    def __init__(self, number_input):
        #Khởi tạo chức năng Random
        np.random.seed(1)

        self.class_two = 4
        self.class_three = 3

        self.weigth_1 = 2 * np.random.random((number_input, self.class_two)) - 1
        self.weigth_2 = 2 * np.random.random((self.class_two, self.class_three)) - 1
        self.weigth_3 = 2 * np.random.random((self.class_three, 1)) - 1
    def __sigmoid(self, Y):
        #Y là hàm tổng
        return 1 / (1 + np.exp(-Y))
    def __sigmoid_derivative(self, Y):
        return Y * (1 - Y)

    def train(self, train_input, train_output, interaction):
        for i in range(interaction):
            #Bước 1: Forward
            self.One = np.dot(train_input, self.weigth_1)
            self.One = self.__sigmoid(self.One)

            self.Two = np.dot(self.One, self.weigth_2)
            self.Two = self.__sigmoid(self.Two)

            self.Out = np.dot(self.Two, self.weigth_3)
            self.Out = self.__sigmoid(self.Out)

            #Bước 2: Tính Lỗi
            Err_Out = (train_output - self.Out) * self.__sigmoid_derivative(self.Out)

            #Tính Error
            Err_Two = np.dot(self.__sigmoid_derivative(self.Two), self.weigth_3)
            Err_Two = np.dot(Err_Two, Err_Out.T)
            Err_One = np.dot(self.__sigmoid_derivative(self.One).T, self.weigth_2)
            Err_One = np.dot(Err_One.T, Err_Two)

            #Bước 3: Cập nhật trọng số
            self.weigth_1 += np.dot(train_input, Err_One.T)
            self.weigth_2 += np.dot(self.Two.T, Err_Two)
            self.weigth_3 += np.dot(self.Out.T, Err_Out)

    def think(self, input):
        print(self.weigth_1)
        One = self.__sigmoid(np.dot(input, self.weigth_1))
        Two = self.__sigmoid(np.dot(One, self.weigth_2))
        return self.__sigmoid(np.dot(Two, self.weigth_3))



NN = mutiplayer_neural_network(4)

training_set_inputs = np.array([[4.7, 3.2, 1.3, 0.2], [6.1, 2.8, 4.7, 1.2], [5.6, 3.0, 4.1, 1.3], [5.8, 2.7, 5.1, 1.9]])
training_set_outputs = np.array([[0, 1, 1, 0]]).T

NN.train(training_set_inputs, training_set_outputs, 10000)
NN.think(NN.think(np.array([6.5, 3.2, 5.1, 2.0])))

