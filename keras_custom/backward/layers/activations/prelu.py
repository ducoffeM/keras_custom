from keras.layers import Layer, PReLU
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardNonLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch

from keras import KerasTensor as Tensor
from .prime import relu_prime


class BackwardPReLU(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `PReLU` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import PReLU
    from keras_custom.backward.layers import BackwardPReLU

    # Assume `activation_layer` is a pre-defined PReLU layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: PReLU,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call(self, inputs, training=None, mask=None):
        layer_output: Tensor = inputs[0]
        layer_input: Tensor = inputs[1]

        reshape_tag, layer_output, n_out = reshape_to_batch(layer_output, list(self.layer.output.shape))
        backward_output: Tensor = relu_prime(
            layer_input,
            negative_slope=self.layer.alpha,
        )
        output = layer_output * backward_output

        if reshape_tag:
            return K.reshape(output, [-1] + n_out + list(self.layer.input.shape)[1:])
        else:
            return output


def get_backward_PReLU(layer: PReLU, use_bias=True) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `PReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `PReLU` layer, using the
    `BackwardPReLU`.

    ### Parameters:
    - `layer`: A Keras `PReLU` layer instance. The function uses this layer's configurations to set up the `BackwardPReLU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardPReLU`, which acts as the reverse layer for the given `PReLU`.

    ### Example Usage:
    ```python
    from keras.layers import PReLU
    from keras_custom.backward import get_backward_PReLU

    # Assume `activation_layer` is a pre-defined PReLU layer
    backward_layer = get_backward_PReLU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardPReLU(layer, use_bias)