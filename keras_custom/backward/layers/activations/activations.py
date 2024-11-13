from keras.layers import Layer, Activation
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardNonLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch

from keras import KerasTensor as Tensor
from .prime import deserialize


class BackwardActivation(BackwardNonLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Activation` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Activation
    from keras_custom.backward.layers import BackwardActivation

    # Assume `activation_layer` is a pre-defined Activation layer
    backward_layer = BackwardActivation(activation_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Activation,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
        activation_name = layer.get_config()["activation"]
        self.layer_backward = deserialize(activation_name)

    def call(self, inputs, training=None, mask=None):
        layer_output: Tensor = inputs[0]
        layer_input: Tensor = inputs[0]

        reshape_tag, layer_output, n_out = reshape_to_batch(layer_output, list(self.layer.output.shape))
        backward_output: Tensor = self.layer_backward(layer_input)
        output = layer_output * backward_output

        if reshape_tag:
            return K.reshape(output, [-1] + n_out + list(self.layer.input.shape)[1:])
        else:
            return output


def get_backward_Activation(layer: Activation, use_bias=True) -> Layer:
    """
    This function creates a `BackwardActivation` layer based on a given `Activation` layer. It provides
    a convenient way to obtain the backward pass of the input `Activation` layer, using the
    `BackwardActivation`.

    ### Parameters:
    - `layer`: A Keras `Activation` layer instance. The function uses this layer's configurations to set up the `BackwardActivation` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardActivation`, which acts as the reverse layer for the given `Activation`.

    ### Example Usage:
    ```python
    from keras.layers import Activation
    from keras_custom.backward import get_backward_Activation

    # Assume `activation_layer` is a pre-defined Activation layer
    backward_layer = get_backward_Activation(activation_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardActivation(layer, use_bias)
