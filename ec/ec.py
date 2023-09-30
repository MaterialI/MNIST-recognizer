import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import get_lenet
from conv_net import convnet_forward
from init_convnet import init_convnet
from scipy.io import loadmat


layers = get_lenet()
params = init_convnet(layers)


def rgbNormalize(image):
    return image / 255


normF = np.vectorize(rgbNormalize)
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


def preprocess_image(image):
    ret, binary_image = cv2.threshold(
        image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, out = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    recognized_numbers = []
    fig, axes = plt.subplots(1, num_labels, figsize=(70, 70))
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 50:
            # Extract the character region and resize to 28x28
            char_region = binary_image[y : y + h, x : x + w]

            axes[i].imshow(char_region, cmap="gray")
            axes[i].axis("off")

            char_region = cv2.resize(char_region, (28, 28))
            char_region = normF(char_region)

            top_pad = (28 - char_region.shape[0]) // 2
            bottom_pad = 28 - char_region.shape[0] - top_pad
            left_pad = (28 - char_region.shape[1]) // 2
            right_pad = 28 - char_region.shape[1] - left_pad

            char_region = cv2.copyMakeBorder(
                char_region,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            char_region = np.reshape(char_region, (784, 1), order="F")
            nCharReg = np.copy(char_region)
            nCharReg = np.reshape(nCharReg, (28, 28))

            layers[0]["batch_size"] = 1
            output = convnet_forward(params, layers, nCharReg, test=True)
            resultingProbs = output[1]
            for number in resultingProbs:
                number[0] = np.sum(number)
            sums = resultingProbs[:, 0]

            recognized_digit = np.argmax(sums)

            recognized_numbers.append(recognized_digit)

    return recognized_numbers


def rgbNormalize(image):
    return image / 255


input_image_path = "../images/image1.jpg"  # change the path to the corresponding file

input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)


normFunc = np.vectorize(rgbNormalize)

recognized_numbers = preprocess_image(input_image)


plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis("off")
print(f"Recognized Numbers: {recognized_numbers}")
plt.show()
