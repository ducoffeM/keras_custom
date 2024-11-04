import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling1D
from keras_custom.backward.layer import BackwardLinearLayer
import keras.ops as K


class BackwardGlobalAveragePooling1D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `GlobalAveragePooling1D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the average pooling
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling1D
    from keras_custom.backward.layers import BackwardGlobalAveragePooling1D

    # Assume `average_pooling_layer` is a pre-defined GlobalAveragePooling1D layer
    backward_layer = BackwardGlobalAveragePooling1D(average_pooling_layer)
    output = backward_layer(input_tensor)
    """

    def call(self, inputs, training=None, mask=None):

        if self.layer.data_format == "channels_first":
            w_in = self.layer.input.shape[-1]
            if self.layer.keepdims:
                return K.repeat(inputs, w_in, -1) / w_in
            else:
                return K.repeat(K.expand_dims(inputs, -1), w_in, -1) / w_in
        else:
            w_in = self.layer.input.shape[1]
            if self.layer.keepdims:
                return K.repeat(inputs, w_in, 1) / w_in
            else:
                return K.repeat(K.expand_dims(inputs, 1), w_in, 1) / w_in


def get_backward_GlobalAveragePooling1D(layer: GlobalAveragePooling1D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardGlobalAveragePooling1D` layer based on a given `GlobalAveragePooling1D` layer. It provides
    a convenient way to obtain a backward approximation of the input `AveragePooling1D` layer, using the
    `BackwardGlobalAveragePooling1D` class to reverse the average pooling operation.

    ### Parameters:
    - `layer`: A Keras `GlobalAveragePooling1D` layer instance. The function uses this layer's configurations to set up the `BackwardGlobalAveragePooling1D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardGlobalAveragePooling1D`, which acts as the reverse layer for the given `GlobalAveragePooling1Dn`.

    ### Example Usage:
    ```python
    from keras.layers import GlobalAveragePooling1D
    from keras_custom.backward import get_backward_GlobalAveragePooling1D

    # Assume `average_layer` is a pre-defined GlobalAveragePooling1D layer
    backward_layer = get_backward_GlobalAveragePooling1D(average_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardGlobalAveragePooling1D(layer)
    return layer_backward
