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

        self.weigth_1 = np.random.random((number_input, self.class_two))
        self.weigth_2 = np.random.random((self.class_two, self.class_three))
        self.weigth_3 = np.random.random((self.class_three, 1))
    def __sigmoid(self, Y):
        #Y là hàm tổng
        return 1 / (1 + np.exp(-Y))
    def __sigmoid_derivative(self, Y):
        return Y * (1 - Y)

    def train(self, train_input, train_output, interaction):
        for i in interaction:
            #Bước 1: Forward
            self.One = np.dot(train_input, self.weigth_1)
            self.One = self.__sigmoid(self.One)

            self.Two = np.dot(self.One, self.weigth_2)
            self.Two = self.__sigmoid(self.Two)

            self.Out = np.dot(self.Two, self.weigth_3)
            self.Out = self.__sigmoid(self.Out)

            #Bước 2: Tính Lỗi

            #Tính lỗi tại Output
            Err_Out = train_output - self.Out

            #Tính Denta

            #Tính lỗi tại One và Two
            Err


