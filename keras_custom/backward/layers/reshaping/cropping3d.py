from keras.layers import ZeroPadding3D, Cropping3D
from keras.layers import Layer
from keras_custom.backward.layers.layer import BackwardLinearLayer


class BackwardCropping3D(BackwardLinearLayer):
    """
    This class implements a custom layer for backward pass of a `Cropping3D` layer in Keras.
    It can be used to apply operations in a reverse manner, reshaping, splitting, and reconstructing the cropping
    outputs back to the original input shape.

    ### Example Usage:
    ```python
    from keras.layers import Cropping3D
    from keras_custom.backward.layers import BackwardCropping3D

    # Assume `cropping_layer` is a pre-defined Cropping3D layer
    backward_layer = BackwardCropping3D(cropping_layer)
    output = backward_layer(input_tensor)
    """

    def __init__(
        self,
        layer: Cropping3D,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        dico_cropping = layer.get_config()
        cropping = dico_cropping["cropping"]
        data_format = dico_cropping["data_format"]

        self.layer_backward = ZeroPadding3D(padding=cropping, data_format=data_format)
        self.layer_backward.built = True

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_Cropping3D(layer: Cropping3D, use_bias=True) -> Layer:
    """
    This function creates a `BackwardCropping2D` layer based on a given `Cropping3D` layer. It provides
    a convenient way to obtain a backward approximation of the input `Cropping2D` layer, using the
    `BackwardCropping3D` class to reverse the cropping operation.

    ### Parameters:
    - `layer`: A Keras `Cropping3D` layer instance. The function uses this layer's configurations to set up the `BackwardCropping3D` layer.
    - `use_bias`: Boolean, optional (default=True). Specifies whether the bias should be included in the
      backward layer.

    ### Returns:
    - `layer_backward`: An instance of `BackwardCropping3D`, which acts as the reverse layer for the given `Cropping3D`.

    ### Example Usage:
    ```python
    from keras.layers import Cropping3D
    from keras_custom.backward import get_backward_Cropping3D

    # Assume `cropping_layer` is a pre-defined Cropping3D layer
    backward_layer = get_backward_Cropping3D(cropping_layer, use_bias=True)
    output = backward_layer(input_tensor)
    """

    return BackwardCropping3D(layer)
