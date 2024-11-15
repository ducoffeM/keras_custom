from keras_custom.backward.layers.layer import BackwardLinearLayer
from keras_custom.backward.layers.utils import reshape_to_batch
from keras.layers import Layer, EinsumDense
from keras.src import ops
import keras.ops as K


class BackwardEinsumDense(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `EinsumDense` layer in Keras.
    It can be used to apply operations in a reverse manner back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import EinsumDense
    from keras_custom.backward.layers import BackwardEinsumDense

    # Assume `dense_layer` is a pre-defined EinsumDense layer
    backward_layer = BackwardEinsumDense(dense_layer)
    output = backward_layer(input_tensor)
    ```
    This code creates an instance of `BackwardEinsumDense` that will apply the transposed
    operation of `original_dense_layer` on the input tensor `inputs`.
    """

    def __init__(
        self,
        layer: EinsumDense,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)

    def call(self, inputs, training=None, mask=None):
        reshape_tag, inputs, n_out = reshape_to_batch(inputs, list(self.layer.output.shape))

        if self.use_bias:
            input_dim_wo_batch = list(self.layer.input.shape[1:])
            x = K.add(inputs, -self.layer(K.zeros([1] + input_dim_wo_batch)))
            x = ops.einsum(self.layer.equation, x, K.transpose(self.layer.kernel))
        else:
            x = ops.einsum(self.layer.equation, inputs, K.transpose(self.layer.kernel))

        if reshape_tag:
            x = K.reshape(x, [-1] + n_out + list(self.layer.input.shape[1:]))

        return x


def get_backward_EinsumDense(layer: EinsumDense, use_bias=True) -> Layer:
    """
    This function creates a `BackwardEinsumDense` layer based on a given `EinsumDense` layer. It provides
    a convenient way to obtain the ackward pass of the input `EinsumDense` layer, using the
    `BackwardEinsumDense` class.

    ### Parameters:
    - `layer`: A Keras `EinsumDense` layer instance. The function uses this layer's configurations to set up the `BackwardEinsumDense` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardEinsumDense`, which acts as the reverse layer for the given `EinsumDense`.

    ### Example Usage:
    ```python
    from keras.layers import EinsumDense
    from keras_custom.backward import get_backward_EinsumDense

    # Assume `dense_layer` is a pre-defined Dense layer
    backward_layer = get_backward_EinsumDense(dense_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """
    return BackwardEinsumDense(layer, use_bias)
