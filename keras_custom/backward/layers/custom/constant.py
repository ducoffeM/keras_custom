from keras_custom.layers import MulConstant, PlusConstant
from keras.layers import Layer
from keras_custom.backward.layers.layer import BackwardLinearLayer


class BackwardMulConstant(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `MulConstant` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import MulConstant
    from keras_custom.backward.layers import BackwardMulConstant

    # Assume `mulconst_layer` is a pre-defined MulConstant layer
    backward_layer = BackwardMulConstant(conv_layer)
    output = backward_layer(input_tensor)
    """
    def __init__(
        self,
        layer: MulConstant,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        self.layer_backward = MulConstant(constant=1.0 / layer.constant)
        self.layer_backward.built = True

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_MulConstant(layer: MulConstant, use_bias=True) -> Layer:
    """
    This function creates a `BackwardMulConstant` layer based on a given `MulConstant` layer. It provides
    a convenient way to obtain the ackward pass of the input `MulConstant` layer, using the
    `BackwardMulConstant` class to reverse the operation.

    ### Parameters:
    - `layer`: A Keras `MulConstant` layer instance. The function uses this layer's configurations to set up the `BackwardMulConstant` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardMulConstant`, which acts as the reverse layer for the given `MulConstant`.

    ### Example Usage:
    ```python
    from keras.layers import MulConstant
    from keras_custom.backward import get_backward_MulConstant

    # Assume `mulconst_layer` is a pre-defined MulConstant layer
    backward_layer = get_backward_MulConstant(mulconst_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    return BackwardMulConstant(layer)


class BackwardPlusConstant(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `PlusConstant` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import PlusConstant
    from keras_custom.backward.layers import BackwardPlusConstant

    # Assume `plusconst_layer` is a pre-defined PlusConstant layer
    backward_layer = BackwardPlusConstant(conv_layer)
    output = backward_layer(input_tensor)
    """
    def __init__(
        self,
        layer: PlusConstant,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
        if self.use_bias:
            self.layer_backward = PlusConstant(constant=-layer.constant * layer.sign, minus=(layer.sign == -1))
        else:
            self.layer_backward = PlusConstant(constant=0.0, minus=(layer.sign == -1))

        self.layer_backward.built = True

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_PlusConstant(layer: PlusConstant, use_bias=True) -> Layer:
    """
    This function creates a `BackwardPlusConstant` layer based on a given `PlusConstant` layer. It provides
    a convenient way to obtain the ackward pass of the input `PlusConstant` layer, using the
    `BackwardPlusConstant` class to reverse the operation.

    ### Parameters:
    - `layer`: A Keras `PlusConstant` layer instance. The function uses this layer's configurations to set up the `BackwardPlusConstant` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardPlusConstant`, which acts as the reverse layer for the given `PlusConstant`.

    ### Example Usage:
    ```python
    from keras.layers import PlusConstant
    from keras_custom.backward import get_backward_PlusConstant

    # Assume `plusconst_layer` is a pre-defined PlusConstant layer
    backward_layer = get_backward_PlusConstant(plusconst_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardPlusConstant(layer, use_bias)
