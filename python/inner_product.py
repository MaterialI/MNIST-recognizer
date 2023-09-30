import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape  # d = input parameters
    n = param["w"].shape[1]

    ###### Fill in the code here ######
    data = np.matmul(np.transpose(param["w"]), input["data"])
    result = data + np.transpose(param["b"])

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": result,  # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    dz = output["diff"]
    param_grad["w"] = np.matmul(input_data["data"], np.transpose(dz))
    param_grad["b"] = np.transpose(np.matmul(dz, np.ones((100, 1))))

    # input_od = np.matmul(dz, output["data"])
    input_od = np.matmul(param["w"], dz)
    return param_grad, input_od
