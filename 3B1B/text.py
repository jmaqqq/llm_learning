from lens import images,labels
import numpy as np
import os
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

folder_name="model_weights"
param_names=['W1','W2','W3','b1','b2','b3']

params={}
for param in param_names:
    file_path=os.path.join(folder_name,f"{param}.npy")
    params[param]=np.load(file_path)
def predict(image_index):

    a0 = images[image_index]
    a1 = sigmoid(np.dot(params['W1'], a0) + params['b1'])
    a2 = sigmoid(np.dot(params['W2'], a1) + params['b2'])
    a3 = sigmoid(np.dot(params['W3'], a2) + params['b3'])

    prediction = np.argmax(a3)
    actual = labels[image_index]

    print(f"网络预测结果: {prediction}")
    print(f"实际正确答案: {actual}")
    if prediction == actual:
        print("识别正确！")
    else:
        print("识别错误。")


predict(50000)