from keras.layers import Reshape
from keras.layers import Layer
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch


class BackwardReshape(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Reshape` layer in Keras.
    It can be used to apply operations in a reverse manner, transposing the permute
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Reshape
    from keras_custom.backward.layers import BackwardReshape

    # Assume `reshape_layer` is a pre-defined Reshape layer
    backward_layer = BackwardReshape(permute_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Reshape,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        input_shape_wo_batch = layer.input.shape[1:]
        self.layer_backward = Reshape(target_shape=input_shape_wo_batch)

    def call(self, inputs, training=None, mask=None):
        reshape_tag, inputs, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))
        output = self.layer_backward(inputs)
        if reshape_tag:
            output = K.reshape(output, [-1]+n_out+list(self.layer.input.shape[1:]))

        return output


def get_backward_Reshape(layer: Reshape, use_bias=True) -> Layer:
    """
    This function creates a `BackwardReshape` layer based on a given `Reshape` layer. It provides
    a convenient way to obtain a backward approximation of the input `Reshape` layer, using the
    `BackwardReshape` class to reverse the reshape operation.

    ### Parameters:
    - `layer`: A Keras `Reshape` layer instance. The function uses this layer's configurations to set up the `BackwardReshape` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardReshape`, which acts as the reverse layer for the given `Reshape`.

    ### Example Usage:
    ```python
    from keras.layers import Reshape
    from keras_custom.backward import get_backward_Reshape

    # Assume `reshape_layer` is a pre-defined Reshape layer
    backward_layer = get_backward_Reshape(reshape_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardReshape(layer)
