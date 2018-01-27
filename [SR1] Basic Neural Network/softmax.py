#Softmax là một hàm

import numpy as np
def softmax_version_1(z):
    z_exp = np.exp(z)
    return z_exp / z_exp.sum()

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
print(softmax_version_1(z))

def softmax_version_2(x):
    x_exp = np.exp(x - np.max(x, axis=0, keepdims=True))
    return x_exp / x_exp.sum()

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
print(softmax_version_2(z))
