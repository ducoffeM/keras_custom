from keras.layers import Layer, ReLU
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardNonLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch

from keras import KerasTensor as Tensor
from .prime import relu_prime


class BackwardReLU(BackwardNonLinearLayer):
    """
    This function creates a `BackwardReLU` layer based on a given `ReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `ReLU` layer, using the
    `BackwardReLU`.

    ### Parameters:
    - `layer`: A Keras `ReLU` layer instance. The function uses this layer's configurations to set up the `BackwardReLU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardReLU`, which acts as the reverse layer for the given `ReLU`.

    ### Example Usage:
    ```python
    from keras.layers import ReLU
    from keras_custom.backward import get_backward_ReLU

    # Assume `activation_layer` is a pre-defined ReLU layer
    backward_layer = get_backward_ReLU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: ReLU,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        backward_relu: Tensor = relu_prime(
            input,
            negative_slope=self.layer.negative_slope,
            threshold=self.layer.threshold,
            max_value=self.layer.max_value,
        )
        output = gradient * backward_relu
        return output


def get_backward_ReLU(layer: ReLU, use_bias=True) -> Layer:
    """
    This function creates a `BackwardELU` layer based on a given `ReLU` layer. It provides
    a convenient way to obtain the backward pass of the input `ReLU` layer, using the
    `BackwardReLU`.

    ### Parameters:
    - `layer`: A Keras `ReLU` layer instance. The function uses this layer's configurations to set up the `BackwardReLU` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardReLU`, which acts as the reverse layer for the given `ReLU`.

    ### Example Usage:
    ```python
    from keras.layers import ReLU
    from keras_custom.backward import get_backward_ReLU

    # Assume `activation_layer` is a pre-defined ReLU layer
    backward_layer = get_backward_ReLU(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardReLU(layer, use_bias)
