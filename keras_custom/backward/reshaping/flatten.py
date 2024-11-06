from keras.layers import Flatten, Reshape, Permute
from keras.layers import Layer
from keras.models import Sequential
import numpy as np
from keras_custom.backward.layer import BackwardLinearLayer


class BackwardFlatten(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Flatten` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, the flattened
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Flatten
    from keras_custom.backward.layers import BackwardFlatten

    # Assume `flatten_layer` is a pre-defined Flatten layer
    backward_layer = BackwardFlatten(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Flatten,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        input_shape_wo_batch = list(layer.input.shape[1:])
        if layer.data_format == "channels_first" and len(input_shape_wo_batch) > 1:
            # we first permute to obtain channel_last format and then flatten
            target_shape = input_shape_wo_batch[1:] + [input_shape_wo_batch[0]]
            layer_reshape = Reshape(target_shape=target_shape)

            dim = [i + 1 for i in range(len(input_shape_wo_batch))]
            dim_permute = dim[1:] + [1]
            dims_backward = np.argsort(dim_permute) + 1
            layer_perm = Permute(dims_backward)
            self.layer_backward = Sequential([layer_reshape, layer_perm])
        else:

            self.layer_backward = Reshape(target_shape=input_shape_wo_batch)

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_Flatten(layer: Flatten, use_bias=True) -> Layer:
    """
    This function creates a `BackwardFlatten` layer based on a given `Flatten` layer. It provides
    a convenient way to obtain a backward approximation of the input `Flatten` layer, using the
    `BackwardFlatten` class to reverse the flatten operation.

    ### Parameters:
    - `layer`: A Keras `Flatten` layer instance. The function uses this layer's configurations to set up the `BackwardFlatten` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardFlatten`, which acts as the reverse layer for the given `Flatten`.

    ### Example Usage:
    ```python
    from keras.layers import Flatten
    from keras_custom.backward import get_backward_Flatten

    # Assume `flatten_layer` is a pre-defined Flatten layer
    backward_layer = get_backward_Flatten(flatten_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    return BackwardFlatten(layer)
