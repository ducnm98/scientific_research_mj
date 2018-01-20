#Code by Michael John
#Thuật toán sử dụng Neural 2 lớp (input và output)
#Thêm thư viện xử lý toán học numpy vào
import numpy as np

class NeuralNetwork():
    def __init__(self):
        #Lớp khởi tạo
        np.random.seed(1)

    def createRandomeWeigth(self, n):
        #Khởi tạo trọng số ngẫu nhiên
        self.weigth =  np.random.random((n, 1))

    def __sigmoid(self, Y):
        #Tính Sigmoid với X là hàm tổng
        return 1 / (1 + np.exp(-Y))

    def __sigmoidWithDerivative(self, Y):
        #Đạo hàm Sigmoid
        return Y * (1 - Y)

    def trainData(self, train_input, train_output, interaction):
        #Hàm training cho Neural network

        #Bước 1: Khởi tạo các trọng số
        self.createRandomeWeigth(len(train_input[0]))

        for inter in range(interaction):
            #Bước 2: Tính tổng của Input và Trọng số
            Y = np.dot(train_input, self.weigth)

            #Bước 3: Sigmoid hàm tổng
            Y = self.__sigmoid(Y)

            #Bước 4: Kiểm tra lỗi
            error = train_output - Y

            #Bước 5: Cập nhật trọng số
            self.weigth += np.dot(train_input.T, error * self.__sigmoidWithDerivative(Y))

    def think(self, input):
        return self.__sigmoid(np.dot(input, self.weigth))


neural_network = NeuralNetwork()

training_set_inputs = np.array([[ 4.7, 3.2, 1.3, 0.2], [ 6.1, 2.8, 4.7, 1.2], [ 5.6, 3.0, 4.1, 1.3], [ 5.8, 2.7, 5.1, 1.9], [ 6.5, 3.2, 5.1, 2.0]])
training_set_outputs = np.array([[0, 1, 1, 0, 0]]).T


neural_network.trainData(training_set_inputs, training_set_outputs, 10000)

print("New synaptic weights after training: ")
print(neural_network.weigth)

print("Considering new situation [1, 0, 0] -> ?: ")
print(neural_network.think(np.array([ 5.8, 2.7, 3.9, 1.2])))

