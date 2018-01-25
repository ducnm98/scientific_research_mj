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
        self.createRandomeWeigth(15)

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

training_set_inputs = np.array([147,150,153,155,158,160,163,165,168,170,173,175,178,180,183])
training_set_outputs = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]).T
training_set_outputs = training_set_outputs / 100


neural_network.trainData(training_set_inputs, training_set_outputs, 10000)

print("New synaptic weights after training: ")
print(neural_network.weigth)

print("Considering new situation [1, 0, 0] -> ?: ")
print(neural_network.think(np.array([155])))

