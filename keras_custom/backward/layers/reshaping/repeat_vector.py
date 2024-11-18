from keras.layers import RepeatVector, Layer
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch


class BackwardRepeatVector(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `RepeatVector` layer in Keras.
    It can be used to apply operations back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import RepeatVector
    from keras_custom.backward.layers import BackwardRepeatVector

    # Assume `repeat_layer` is a pre-defined RepeatVector layer
    backward_layer = BackwardRepeatVector(repeat_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: RepeatVector,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)

        self.layer = layer

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        return K.max(gradient, axis=1)


def get_backward_RepeatVector(layer: BackwardRepeatVector, use_bias=True) -> Layer:
    """
    This function creates a `BackwardRepeatVector` layer based on a given `RepeatVector` layer. It provides
    a convenient way to obtain a backward approximation of the input `RepeatVector` layer, using the
    `BackwardRepeatVector` class to reverse the repeat operation.

    ### Parameters:
    - `layer`: A Keras `RepeatVector` layer instance. The function uses this layer's configurations to set up the `BackwardRepeatVector` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardRepeatVector`, which acts as the reverse layer for the given `RepeatVector`.

    ### Example Usage:
    ```python
    from keras.layers import RepeatVector
    from keras_custom.backward import get_backward_RepeatVector

    # Assume `repeat_layer` is a pre-defined RepeatVector layer
    backward_layer = get_backward_RepeatVector(repeat_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    layer_backward = BackwardRepeatVector(layer)
    return layer_backward
