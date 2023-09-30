import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image, ImageOps
import cv2
import torch

# print(torch.__version__)
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU is not available; using CPU")

def rgbNormalize(image):
    return image / 255


normF = np.vectorize(rgbNormalize)

layers = get_lenet()
params = init_convnet(layers)
layers[0]["batch_size"] = 1

data = loadmat("../results/lenet.mat")
params_raw = data["params"]

for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    assert (
        params[params_idx]["w"].shape == raw_w.shape
    ), "weights do not have the same shape"
    assert (
        params[params_idx]["b"].shape == raw_b.shape
    ), "biases do not have the same shape"
    params[params_idx]["w"] = raw_w
    params[params_idx]["b"] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)

imageData = []
data_folder = ".." + "\data" + chr(92) + "test"
# Loop through the files in the folder
numbers = []
count = 0
for filename in os.listdir(data_folder):
    number = ""
    number = filename[len(filename) - 1]
    af = data_folder + chr(92) + number
    for file in os.listdir(af):
        img = cv2.imread(os.path.join(af, file), cv2.IMREAD_GRAYSCALE)
        img = normF(img)
        img = np.reshape(np.array(img), (28, 28))

        imageData = np.append(imageData, img)

        numbers = np.append(numbers, (int)(ord(number) - ord("0")))
        count += 1
imageData = np.reshape(imageData, (count, 28, 28))

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xvalidate = np.array(xvalidate)
yvalidate = np.array(yvalidate)
xtest = np.array(imageData)
ytest = np.array(numbers)
pr = np.zeros((xtest.size, 2))

# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
for i in range(0, count):
    cptest, P = convnet_forward(params, layers, xtest[i], test=True)
    all_preds.extend(np.argmax(P, axis=0))

confusion = confusion_matrix(np.ndarray.flatten(ytest), all_preds)

print("Confusion Matrix:")
print(confusion)

print("Classification Report:")
print(classification_report(np.ndarray.flatten(ytest), all_preds))
# hint:
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)
