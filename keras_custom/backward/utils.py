from typing import List, Union
from keras.layers import ZeroPadding2D, Cropping2D, ZeroPadding1D, Cropping1D, ZeroPadding3D, Cropping3D
import keras.ops as K


# compute output shape post convolution
def compute_output_pad(input_shape_wo_batch, kernel_size, strides, padding, data_format):
    if data_format == "channels_first":
        w, h = input_shape_wo_batch[1:]
    else:
        w, h = input_shape_wo_batch[:-1]
    k_w, k_h = kernel_size
    if padding == "same":
        p = 0
    s_w, s_h = strides

    w_pad = (w - k_w + 2 * p) / s_w + 1 - w
    h_pad = (h - k_h + 2 * p) / s_h + 1 - h
    return (w_pad, h_pad)


def pooling_layer2D(w_pad, h_pad, data_format) -> List[Union[ZeroPadding2D, Cropping2D]]:
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


def pooling_layer1D(w_pad, data_format) -> List[Union[ZeroPadding1D, Cropping1D]]:
    if w_pad:
        # add padding
        if w_pad >= 0:
            padding = (w_pad // 2, w_pad // 2 + w_pad % 2)
            pad_layer = [ZeroPadding1D(padding)]
        else:
            w_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (w_pad // 2, w_pad // 2 + w_pad % 2)
            pad_layer = [Cropping1D(cropping)]
        return pad_layer
    return []


def pooling_layer3D(d_pad, w_pad, h_pad, data_format) -> List[Union[ZeroPadding3D, Cropping3D]]:
    if d_pad or w_pad or h_pad:
        # add padding
        if d_pad >= 0 and w_pad >= 0 and h_pad >= 0:
            padding = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [ZeroPadding3D(padding, data_format=data_format)]
        elif d_pad <= 0 and w_pad <= 0 and h_pad <= 0:
            d_pad *= -1
            w_pad *= -1
            h_pad *= -1
            # padding = ((0, -w_pad), (0, -h_pad))
            cropping = (
                (d_pad // 2, d_pad // 2 + d_pad % 2),
                (w_pad // 2, w_pad // 2 + w_pad % 2),
                (h_pad // 2, h_pad // 2 + h_pad % 2),
            )
            pad_layer = [Cropping3D(cropping, data_format=data_format)]
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
    else:
        return []


def call_backward_depthwise2d(inputs, layer, op_reshape, op_split, inner_models, axis, axis_c, c_in, use_bias):
    # remove bias if needed
    if hasattr(layer, "use_bias") and layer.use_bias and use_bias:
        if layer.data_format == "channels_first":
            inputs = inputs - layer.bias[None, :, None, None]  # (batch, d_m*c_in, w_out, h_out)
        else:
            inputs = inputs - layer.bias[None, None, None, :]  # (batch, w_out, h_out, d_m*c_in)

    outputs = op_reshape(inputs)  # (batch, d_m, c_in, w_out, h_out) if data_format=channel_first

    # if self.layer.use_bias and self.use_bias:

    split_outputs = K.split(outputs, c_in, axis=axis_c)  # [(batch, d_m, 1, w_out, h_out)]
    split_outputs = [op_split(s_o_i) for s_o_i in split_outputs]  # [(batch_size, d_m, w_out, h_out)]

    conv_outputs = [inner_models[i](s_o_i) for (i, s_o_i) in enumerate(split_outputs)]  # [(batch_size, 1, w_in, h_in)]
    return K.concatenate(conv_outputs, axis=axis)  # (batch_size, c_in, w_in, h_in)


"""
def init_backward_depthwise2d(layer, depth_multiplier, kernel):
    dico_attributes=dict()

    input_dim_wo_batch = layer.input.shape[1:]
    output_dim_wo_batch = layer.output.shape[1:]

    d_m = depth_multiplier
    if layer.data_format == "channels_first":
            c_in = input_dim_wo_batch[0]
            w_out, h_out = output_dim_wo_batch[-2:]
            #target_shape = [self.layer.depth_multiplier, c_in, w_out, h_out]
            target_shape = [c_in, d_m, w_out, h_out]

            split_shape = [d_m, w_out, h_out]
            axis = 1
            c_in = c_in
            axis_c = 1
    else:
            c_in = input_dim_wo_batch[-1]
            w_out, h_out = output_dim_wo_batch[:2]
            #target_shape = [w_out, h_out, c_in, self.layer.depth_multiplier]
            target_shape = [w_out, h_out, d_m, c_in]

            split_shape = [w_out, h_out, d_m]
            axis = -1
            c_in = c_in
            #self.axis_c = -2
            axis_c = -1

        op_reshape = Reshape(target_shape)
        op_split = Reshape(split_shape)

        # c_in convolution operator

        conv_transpose_list: List[Conv2DTranspose] = []

        for i in range(c_in):
            kernel_i = kernel[:, :, i : i + 1]  # (kernel_w, kernel_h, 1, d_m)
            dico_depthwise_conv = layer.get_config()
            dico_depthwise_conv["filters"] = dico_depthwise_conv["depth_multiplier"]
            dico_depthwise_conv["kernel_initializer"] = dico_depthwise_conv["depthwise_initializer"]
            dico_depthwise_conv["kernel_regularizer"] = dico_depthwise_conv["depthwise_regularizer"]
            dico_depthwise_conv["kernel_constraint"] = dico_depthwise_conv["depthwise_constraint"]
            dico_depthwise_conv["padding"] = "valid" #self.layer.padding

            # remove unknown features in Conv2DTranspose
            dico_depthwise_conv.pop("depth_multiplier")
            dico_depthwise_conv.pop("depthwise_initializer")
            dico_depthwise_conv.pop("depthwise_regularizer")
            dico_depthwise_conv.pop("depthwise_constraint")

            dico_depthwise_conv["use_bias"] = False

            conv_t_i = Conv2DTranspose.from_config(dico_depthwise_conv)
            conv_t_i.kernel = kernel_i
            conv_t_i.built = True
            conv_transpose_list.append(conv_t_i)

        # shape of transposed input
        input_dim_wo_batch_t = (K.repeat(conv_t_i(K.zeros([1] + split_shape)), c_in, axis=self.axis)[0]).shape
        if self.layer.data_format == "channels_first":
            w_pad = input_dim_wo_batch[-2] - input_dim_wo_batch_t[-2]
            h_pad = input_dim_wo_batch[-1] - input_dim_wo_batch_t[-1]
        else:
            w_pad = input_dim_wo_batch[0] - input_dim_wo_batch_t[0]
            h_pad = input_dim_wo_batch[1] - input_dim_wo_batch_t[1]

        pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            self.inner_models = [Sequential([conv_t_i] + pad_layers) for conv_t_i in conv_transpose_list]
        else:
            self.inner_models = conv_transpose_list
"""
