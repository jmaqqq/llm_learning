import os
import numpy as np
from lens import images,labels

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_derivative(a):
    return a * (1.0 - a)

W1 = np.random.randn(16, 784)
b1 = np.random.randn(16, 1)
W2 = np.random.randn(16, 16)
b2 = np.random.randn(16, 1)
W3 = np.random.randn(10, 16)
b3 = np.random.randn(10, 1)

learning_rate = 0.1
print("开始训练...")

for i in range(len(images)):
    a0 = images[i]
    target = labels[i]

    y = np.zeros((10, 1))
    y[target] = 1.0

    z1 = np.dot(W1, a0) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    output_error = 2 * (a3 - y)
    delta3 = output_error * sigmoid_derivative(a3)
    delta2 = np.dot(W3.T, delta3) * sigmoid_derivative(a2)
    delta1 = np.dot(W2.T, delta2) * sigmoid_derivative(a1)

    W3 -= learning_rate * np.dot(delta3, a2.T)
    b3 -= learning_rate * delta3
    W2 -= learning_rate * np.dot(delta2, a1.T)
    b2 -= learning_rate * delta2
    W1 -= learning_rate * np.dot(delta1, a0.T)
    b1 -= learning_rate * delta1

    if i % 1000 == 0:
        current_cost = np.sum((a3 - y) ** 2)
        print(f"图片索引: {i}, 当前代价(Cost): {current_cost:.4f}")
print("训练完成！")

folder_name="model_weights"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
np.save(os.path.join(folder_name, "W1.npy"), W1)
np.save(os.path.join(folder_name, "b1.npy"), b1)
np.save(os.path.join(folder_name, "W2.npy"), W2)
np.save(os.path.join(folder_name, "b2.npy"), b2)
np.save(os.path.join(folder_name, "W3.npy"), W3)
np.save(os.path.join(folder_name, "b3.npy"), b3)



