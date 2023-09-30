import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
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
m_train = xtrain.shape[1]

batch_size = 1
layers[0]["batch_size"] = batch_size

img = xtest[:, 0]
img = np.reshape(img, (28, 28), order="F")
plt.imshow(img.T, cmap="gray")
plt.show()

output = convnet_forward(params, layers, xtest[:, 0:1])
output_1 = np.reshape(output[0]["data"], (28, 28), order="F")

conv2_activations = output[1]["data"]
relu3_activations = output[3]["data"]

# Reshape activations for plotting
conv2_activations = np.reshape(
    conv2_activations, (24, 24, 20), order="F"
)  # Assuming 32x32 activations for CONV2
relu3_activations = np.reshape(
    relu3_activations, (12, 12, 20), order="F"
)  # Assuming 16x16 activations for RELU3

# Create a figure to display the activations
plt.figure(figsize=(12, 6))
plt.suptitle("Visualizing Activations of CONV2 and RELU3 Layers")

# Plot 20 features from CONV2 layer
for i in range(20):
    plt.subplot(4, 10, i + 1)
    plt.imshow(conv2_activations[:, :, i], cmap="Greys")
    plt.axis("off")
    plt.title(f"CONV2 Feature {i + 1}")

# Plot 20 features from RELU3 layer
for i in range(20):
    plt.subplot(4, 10, i + 21)
    plt.imshow(relu3_activations[:, :, i], cmap="Greys")
    plt.axis("off")
    plt.title(f"RELU3 Feature {i + 1}")

plt.show()
print("FFFFFFFFFFUCCCCCCCCK")
