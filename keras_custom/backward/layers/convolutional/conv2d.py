from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer
from keras.models import Sequential
from keras_custom.backward.layers.utils import compute_output_pad, pooling_layer2D
from keras_custom.backward.layers.layer import BackwardLinearLayer


class BackwardConv2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Conv2D` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Conv2D
    from keras_custom.backward.layers import BackwardConv2D

    # Assume `conv_layer` is a pre-defined Conv2D layer
    backward_layer = BackwardConv2D(conv_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Conv2D,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
        dico_conv = layer.get_config()
        dico_conv.pop("groups")
        input_shape = list(layer.input.shape[1:])
        # update filters to match input, pay attention to data_format
        if layer.data_format == "channels_first":  # better to use enum than raw str
            dico_conv["filters"] = input_shape[0]
        else:
            dico_conv["filters"] = input_shape[-1]

        dico_conv["use_bias"] = use_bias
        dico_conv["padding"] = "valid"

        layer_backward = Conv2DTranspose.from_config(dico_conv)
        layer_backward.kernel = layer.kernel
        if use_bias:
            layer_backward.bias = layer.bias

        layer_backward.built = True

        input_shape_wo_batch = list(layer.input.shape[1:])
        input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)

        if layer.data_format == "channels_first":
            w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
            h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
        else:
            w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
            h_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]

        pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
        if len(pad_layers):
            layer_backward = Sequential([layer_backward] + pad_layers)
            _ = layer_backward(layer.output)
        self.layer_backward = layer_backward

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_Conv2D(layer: Conv2D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardConv2D` layer based on a given `Conv2D` layer. It provides
    a convenient way to obtain the ackward pass of the input `Conv2D` layer, using the
    `BackwardConv2D` class to reverse the convolution operation.

    ### Parameters:
    - `layer`: A Keras `Conv2D` layer instance. The function uses this layer's configurations to set up the `BackwardConv2D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardConv2D`, which acts as the reverse layer for the given `Conv2D`.

    ### Example Usage:
    ```python
    from keras.layers import Conv2D
    from keras_custom.backward import get_backward_Conv2D

    # Assume `conv_layer` is a pre-defined Conv2D layer
    backward_layer = get_backward_Conv2D(conv_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardConv2D(layer, use_bias)
