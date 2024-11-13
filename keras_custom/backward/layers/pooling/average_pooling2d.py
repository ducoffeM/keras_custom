import numpy as np
import keras
from keras.layers import AveragePooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, Layer
from keras.models import Sequential
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardLinearLayer
from keras_custom.backward.layers.utils import pooling_layer2D, reshape_to_batch


class BackwardAveragePooling2D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `AveragePooling2D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling2D
    from keras_custom.backward.layers import BackwardAveragePooling2D

    # Assume `average_pooling_layer` is a pre-defined AveragePooling2D layer
    backward_layer = BackwardAveragePooling2D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: AveragePooling2D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

        if not self.layer.built:
            raise ValueError("layer {} is not built".format(layer.name))
        
        if self.layer.padding == "same":
            raise UserWarning("Padding same with AveragePooling is not reperoductible in pure keras operations. See github issue https://github.com/mil-tokyo/webdnn/issues/694")
        

        # average pooling is a depthwise convolution
        # we use convtranspose to invert the convolution of kernel ([1/n..1/n]..[1/n..1/n]) with n the pool size
        pool_size = list(layer.pool_size)
        layer_t = Conv2DTranspose(
            1,
            layer.pool_size,
            strides=layer.strides,
            padding="valid",
            data_format=layer.data_format,
            use_bias=False,
            trainable=False,
        )
        kernel_ = np.ones(pool_size + [1, 1]) / np.prod(pool_size)
        layer_t.kernel = keras.Variable(kernel_)
        layer_t.built = True

        # shape of transposed input
        input_shape_t = list(layer_t(self.layer.output).shape[1:])
        input_shape = list(self.layer.input.shape[1:])

        if layer.data_format == "channels_first":
            w_pad = input_shape[-2] - input_shape_t[-2]
            h_pad = input_shape[-1] - input_shape_t[-1]
        else:
            w_pad = input_shape[0] - input_shape_t[0]
            h_pad = input_shape[1] - input_shape_t[1]

        pad_layers = pooling_layer2D(w_pad, h_pad, self.layer.data_format)

        if len(pad_layers):
            self.model = Sequential([layer_t] + pad_layers)
        else:
            self.model = layer_t
        self.model(self.layer.output)
        self.model.trainable = False
        self.model.built = True

    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    def call(self, inputs, training=None, mask=None):

        reshape_tag, inputs, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))
        # inputs (batch, channel_out, w_out, h_out)
        if self.layer.data_format == "channels_first":
            channel_out = inputs.shape[1]
            axis = 1
        else:
            channel_out = inputs.shape[-1]
            axis = -1

        split_inputs = K.split(inputs, channel_out, axis)
        # apply conv transpose on every of them
        outputs = K.concatenate([self.model(input_i) for input_i in split_inputs], axis)
        if reshape_tag:
            outputs = K.reshape(outputs, [-1]+n_out+list(self.layer.input.shape[1:]))

        return outputs



def get_backward_AveragePooling2D(layer: AveragePooling2D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardAveragePooling2D` layer based on a given `AveragePooling2D` layer. It provides
    a convenient way to obtain a backward approximation of the input `AveragePooling2D` layer, using the
    `BackwardAveragePooling2D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `AveragePooling2D` layer instance. The function uses this layer's configurations to set up the `BackwardAveragePooling2D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardAveragePooling2D`, which acts as the reverse layer for the given `AveragePooling2D`.

    ### Example Usage:
    ```python
    from keras.layers import AveragePooling2D
    from keras_custom.backward import get_backward_AveragePooling2D

    # Assume `average_layer` is a pre-defined AveragePooling2D layer
    backward_layer = get_backward_AveragePooling2D(average_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardAveragePooling2D(layer)
    return layer_backward
