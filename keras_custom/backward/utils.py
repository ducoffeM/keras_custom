
from typing import List, Union
from keras.layers import ZeroPadding2D, Cropping2D

# compute output shape post convolution
def compute_output_pad(input_shape_wo_batch, kernel_size, strides, padding, data_format):
    if data_format=="channels_first":
        w, h = input_shape_wo_batch[1:]
    else:
        w, h = input_shape_wo_batch[:-1]
    k_w, k_h = kernel_size
    if padding =='same':
        p = 0
    s_w, s_h = strides

    w_pad = (w - k_w + 2*p)/s_w +1 - w
    h_pad = (h - k_h + 2*p)/s_h +1 - h
    return (w_pad, h_pad)

def pooling_layer2D(w_pad, h_pad, data_format)->List[Union[ZeroPadding2D, Cropping2D]]:
    if w_pad or h_pad:
        # add padding
        if w_pad >= 0 and h_pad >= 0:
                padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [ZeroPadding2D(padding, data_format=data_format)]
        elif w_pad <= 0 and h_pad <= 0:
                w_pad *= -1
                h_pad *= -1
                # padding = ((0, -w_pad), (0, -h_pad))
                cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [Cropping2D(cropping, data_format=data_format)]
        elif w_pad > 0 and h_pad < 0:
                h_pad *= -1
                padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
                cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [
                    ZeroPadding2D(padding, data_format=data_format),
                    Cropping2D(cropping, data_format=data_format),
                ]
        else:
                w_pad *= -1
                padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
                cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
                pad_layer = [
                    ZeroPadding2D(padding, data_format=data_format),
                    Cropping2D(cropping, data_format=data_format),
                ]
        return pad_layer
    return []