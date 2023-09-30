import numpy as np
from utils import im2col_conv_batch, im2col_conv

images = np.array(
    [
        [1, 0],
        [5, 0],
        [9, 0],
        [13, 0],
        [2, 0],
        [6, 0],
        [10, 0],
        [14, 0],
        [3, 0],
        [7, 0],
        [11, 0],
        [15, 0],
        [4, 0],
        [8, 0],
        [12, 0],
        [16, 0],
        [101, 0],
        [105, 0],
        [109, 0],
        [113, 0],
        [102, 0],
        [106, 0],
        [110, 0],
        [114, 0],
        [103, 0],
        [107, 0],
        [111, 0],
        [115, 0],
        [104, 0],
        [108, 0],
        [112, 0],
        [116, 0],
    ]
)


h_in = 4
w_in = 4
pad = 0
stride = 1
k = 2

h_out = (h_in + 2 * pad - k) // stride + 1
w_out = (w_in + 2 * pad - k) // stride + 1
channel = 2

res = im2col_conv_batch(
    {"height": h_in, "width": w_in, "batch_size": 2, "channel": 2, "data": images},
    {"k": k, "pad": pad, "stride": stride},
    h_out,
    w_out,
)

# print(res.shape)
#
# print("\n")
