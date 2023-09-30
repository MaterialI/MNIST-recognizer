import numpy as np
from utils import im2col_conv, col2im_conv, im2col_conv_batch


def conv_layer_forward(input_data, layer, param):
    """
    Forward pass for a convolutional layer.

    Parameters:
    - input_data (dict): A dictionary containing the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.
    """
    h_in = input_data["height"]
    w_in = input_data["width"]
    c = input_data["channel"]
    batch_size = input_data["batch_size"]
    k = layer["k"]
    pad = layer["pad"]
    stride = layer["stride"]
    num = layer["num"]

    # resolve output shape
    h_out = (h_in + 2 * pad - k) // stride + 1
    w_out = (w_in + 2 * pad - k) // stride + 1

    assert h_out == int(h_out), "h_out is not integer"
    assert w_out == int(w_out), "w_out is not integer"

    input_n = {
        "height": h_in,
        "width": w_in,
        "channel": c,
        "data": input_data["data"],
        "batch_size": batch_size,
    }
    # res = im2col_conv_batch(
    #     input_n, layer, h_out, w_out
    # )  # assume we mult w[i,j] to sample[i,j] over all samples to produce 3 channels

    # tmp = []
    # for i in range(0, c):  # subdivide the input data in 3 subchannels
    #     tmp.append(res[i * (int)(len(res) / c) : (i + 1) * (int)(len(res) / c)])
    # if c != 1:
    #     res = tmp
    # else:  # if only 1 channel prepare the data for parse in the next for loop which parses every input channel separately and produces (r,g,b) for its channel
    #     a = []
    #     a.append(res)
    #     res = np.asarray(a)

    # out = []
    # sum = 0
    # i = 0
    # for inputSample in res:  # parse the number of input channels
    #     for ch in range(
    #         0, num
    #     ):  # make the calculations for every single output channel
    #         for (
    #             sample
    #         ) in (
    #             inputSample
    #         ):  # take a sample from padded input data in the specified input channel and produce the resulting product
    #             subsL = (int)(param["w"].shape[0])
    #             weightBatch = param["w"][:, ch][
    #                 (i * (int)(subsL / c)) : ((i + 1) * (int)(subsL / c))
    #             ]
    #             sum = np.matmul(np.transpose(sample), weightBatch)
    #             sum += param["b"][0][ch]
    #             out.append(sum)
    #     i += 1

    # # due to the number of input channels, we perform component-wise addition for input layers (1,2,3) each of which has (r,g,b) output
    # # here we combine them r = (r1+r2+r3) g = (g1+g2+g3) b= (b1+ b2+b3)
    # rs = out[0 : (int)(len(out) / c)]
    # if c != 1:
    #     for inCh in range(1, c):
    #         for i in range(0, len(rs)):
    #             rs[i] += out[
    #                 inCh * (int)(len(out) / c) : (inCh + 1) * (int)(len(out) / c)
    #             ][i]

    # rs = np.asarray(rs)
    outD = np.zeros((h_out * w_out * num, batch_size))
    output = {
        "height": h_out,
        "width": w_out,
        "channel": num,
        "batch_size": batch_size,
        "data": outD,  # replace 'data' value with your implementation
    }
    data = im2col_conv_batch(input_n, layer, h_out, w_out)
    for batch in range(0, batch_size):
        sample = data[:, :, batch]
        result = np.matmul(np.transpose(sample), param["w"])
        result += param["b"]
        outD[:, batch] = result.flatten("F")

    # tmp = input_data.copy()
    # for b in range(0, batch_size):
    #     tmp["data"] = input_data["data"][:, b].copy()
    #     img = im2col_conv(tmp, layer, h_out, w_out)
    #     img = img.reshape(c * k * k, h_out * w_out)
    #     res = np.matmul(np.transpose(img), param["w"])
    #     res += param["b"]
    #     res = res.reshape(h_out, w_out, num)
    #     output["data"][:, b] = res.reshape(h_out * w_out * num)
    # tmp = []
    # for ch in range(0, num):
    #     for i in range(0, h_out * w_out):
    #         tmp.append(output["data"][ch + i * num])
    # output["data"] = tmp
    output["data"] = outD

    ############# Fill in the code here ###############
    # Hint: use im2col_conv_batch for faster computation

    return output


def conv_layer_backward(output, input_data, layer, param):
    """
    Compute the backward pass for the convolution layer.

    Parameters:
    - output (dict): A dictionary containing the output of the forward pass.
    - input_data (dict): A dictionary containing the original input to the forward function.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.

    Returns:
    - param_grad (dict): A dictionary containing the gradients with respect to the parameters 'b' and 'w'.
    - input_od (numpy.ndarray): The gradients with respect to the input.
    """

    h_in = input_data["height"]
    w_in = input_data["width"]
    c = input_data["channel"]
    batch_size = input_data["batch_size"]
    k = layer["k"]
    group = layer["group"]
    num = layer["num"]

    h_out = output["height"]
    w_out = output["width"]
    input_n = {"height": h_in, "width": w_in, "channel": c}

    input_od = np.zeros(input_data["data"].shape)
    param_grad = {"b": np.zeros(param["b"].shape), "w": np.zeros(param["w"].shape)}

    for n in range(batch_size):
        input_n["data"] = input_data["data"][:, n]
        col = im2col_conv(input_n, layer, h_out, w_out)
        col = np.reshape(col, (k * k * c, h_out * w_out), order="F")
        col_diff = np.zeros(col.shape)
        temp_data_diff = np.reshape(
            output["diff"][:, n], (h_out * w_out, num), order="F"
        )

        for g in range(group):
            g_c_idx = slice(g * k * k * c // group, (g + 1) * k * k * c // group)
            g_num_idx = slice(g * num // group, (g + 1) * num // group)
            col_g = col[g_c_idx, :]
            weight = param["w"][:, g_num_idx]

            # get the gradient of param
            param_grad["b"][:, g_num_idx] += np.sum(
                temp_data_diff[:, g_num_idx], axis=0
            )
            param_grad["w"][:, g_num_idx] += col_g.dot(temp_data_diff[:, g_num_idx])
            col_diff[g_c_idx, :] = weight.dot(temp_data_diff[:, g_num_idx].T)

        im = col2im_conv(col_diff.ravel(order="F"), input_data, layer, h_out, w_out)
        # set the gradient w.r.t to input.data
        input_od[:, n] = im.ravel(order="F")

    return param_grad, input_od
