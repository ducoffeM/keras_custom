from keras.layers import Layer, DepthwiseConv2D, Conv2DTranspose, Reshape, ZeroPadding2D, Cropping2D
from keras_custom.backward.layer import BackwardLinearLayer
from keras.models import Sequential
import keras.ops as K

from typing import List


class BackwardDepthwiseConv2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `DepthwiseConv2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the depthwise convolution
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv2D
    from keras_custom.backward.layers import BackwardDepthwiseConv2D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv2D layer
    backward_layer = BackwardDepthwiseConv2D(depthwise_conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: DepthwiseConv2D,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

        input_dim_wo_batch = self.layer.input.shape[1:]
        output_dim_wo_batch = self.layer.output.shape[1:]
        self.d_m = self.layer.depth_multiplier
        if self.layer.data_format == "channels_first":
            c_in = input_dim_wo_batch[0]
            w_out, h_out = output_dim_wo_batch[-2:]
            target_shape = [self.layer.depth_multiplier, c_in, w_out, h_out]

            split_shape = [self.layer.depth_multiplier, w_out, h_out]
            self.axis = 1
            self.c_in = c_in
            self.axis_c = 2
        else:
            c_in = input_dim_wo_batch[-1]
            w_out, h_out = output_dim_wo_batch[:2]
            target_shape = [w_out, h_out, c_in, self.layer.depth_multiplier]
            split_shape = [w_out, h_out, self.layer.depth_multiplier]
            self.axis = -1
            self.c_in = c_in
            self.axis_c = -2

        self.op_reshape = Reshape(target_shape)
        self.op_split = Reshape(split_shape)

        # c_in convolution operator

        conv_transpose_list: List[Conv2DTranspose] = []

        for i in range(c_in):
            kernel_i = self.layer.kernel[:, :, i : i + 1]  # (kernel_w, kernel_h, 1, d_m)
            dico_depthwise_conv = layer.get_config()
            dico_depthwise_conv["filters"] = dico_depthwise_conv["depth_multiplier"]
            dico_depthwise_conv["kernel_initializer"] = dico_depthwise_conv["depthwise_initializer"]
            dico_depthwise_conv["kernel_regularizer"] = dico_depthwise_conv["depthwise_regularizer"]
            dico_depthwise_conv["kernel_constraint"] = dico_depthwise_conv["depthwise_constraint"]
            dico_depthwise_conv["padding"] = "valid"

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

        if w_pad or h_pad:
            # add padding
            if w_pad >= 0 and h_pad >= 0:
                padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [ZeroPadding2D(padding, data_format=self.layer.data_format)]
            elif w_pad <= 0 and h_pad <= 0:
                w_pad *= -1
                h_pad *= -1
                # padding = ((0, -w_pad), (0, -h_pad))
                cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [Cropping2D(cropping, data_format=self.layer.data_format)]
            elif w_pad > 0 and h_pad < 0:
                h_pad *= -1
                padding = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
                cropping = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
                pad_layer = [
                    ZeroPadding2D(padding, data_format=self.layer.data_format),
                    Cropping2D(cropping, data_format=self.layer.data_format),
                ]
            else:
                w_pad *= -1
                padding = ((0, 0), (h_pad // 2, h_pad // 2 + h_pad % 2))
                cropping = ((w_pad // 2, w_pad // 2 + w_pad % 2), (0, 0))
                pad_layer = [
                    ZeroPadding2D(padding, data_format=self.layer.data_format),
                    Cropping2D(cropping, data_format=self.layer.data_format),
                ]
            self.inner_models = [Sequential([conv_t_i] + pad_layer) for conv_t_i in conv_transpose_list]
        else:
            self.inner_models = conv_transpose_list


    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    # serialize ...

    def call(self, inputs, training=None, mask=None):

        # remove bias if needed
        if self.layer.use_bias and self.use_bias:
            if self.layer.data_format == "channels_first":
                inputs = inputs - self.layer.bias[None, :, None, None]  # (batch, d_m*c_in, w_out, h_out)
            else:
                inputs = inputs - self.layer.bias[None, None, None, :]  # (batch, w_out, h_out, d_m*c_in)

        outputs = self.op_reshape(inputs)  # (batch, d_m, c_in, w_out, h_out) if data_format=channel_first

        # if self.layer.use_bias and self.use_bias:

        split_outputs = K.split(outputs, self.c_in, axis=self.axis_c)  # [(batch, d_m, 1, w_out, h_out)]
        split_outputs = [self.op_split(s_o_i) for s_o_i in split_outputs]  # [(batch_size, d_m, w_out, h_out)]

        conv_outputs = [
            self.inner_models[i](s_o_i) for (i, s_o_i) in enumerate(split_outputs)
        ]  # [(batch_size, 1, w_in, h_in)]
        return K.concatenate(conv_outputs, axis=self.axis)  # (batch_size, c_in, w_in, h_in)


def get_backward_DepthwiseConv2D(layer: DepthwiseConv2D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardDepthwiseConv2D` layer based on a given `DepthwiseConv2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `DepthwiseConv2D` layer, using the
    `BackwardDepthwiseConv2D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `DepthwiseConv2D` layer instance. The function uses this layer's configurations (input and output shapes,
      depth multiplier, data format) to set up the `BackwardDepthwiseConv2D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardDepthwiseConv2D`, which acts as the reverse layer for the given `DepthwiseConv2D`.

    ### Example Usage:
    ```python
    from keras.layers import DepthwiseConv2D
    from keras_custom.backward import get_backward_DepthwiseConv2D

    # Assume `depthwise_conv_layer` is a pre-defined DepthwiseConv2D layer
    backward_layer = get_backward_DepthwiseConv2D(depthwise_conv_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    layer_backward = BackwardDepthwiseConv2D(layer, use_bias)
    return layer_backward
