from keras.layers import Layer, LeakyReLU
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardNonLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch

from keras import KerasTensor as Tensor
from .prime import leaky_relu_prime


class BackwardLeakyReLU(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `LeakyReLU` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import LeakyReLU
    from keras_custom.backward.layers import BackwardLeakyReLU

    # Assume `activation_layer` is a pre-defined LeakyReLU layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: LeakyReLU,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        backward_output: Tensor = leaky_relu_prime(input, negative_slope=self.layer.negative_slope)
        output = gradient * backward_output
        return output


def get_backward_LeakyReLU(layer: LeakyReLU, use_bias=True) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `LeakyReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `LeakyReLU` layer, using the
    `BackwardLeakyReLU`.

    ### Parameters:
    - `layer`: A Keras `LeakyReLU` layer instance. The function uses this layer's configurations to set up the `BackwardLeakyReLU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardLeakyReLU`, which acts as the reverse layer for the given `LeakyReLU`.

    ### Example Usage:
    ```python
    from keras.layers import LeakyReLU
    from keras_custom.backward import get_backward_LeakyReLU

    # Assume `activation_layer` is a pre-defined LeakyReLU layer
    backward_layer = get_backward_LeakyReLU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardLeakyReLU(layer, use_bias)
