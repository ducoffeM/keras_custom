# abstract class for BackwardLayer
from keras.layers import Layer
import keras
import numpy as np

class BackwardLinearLayer(Layer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32), use_bias=False)
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """

    def __init__(
        self,
        layer: Layer,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.use_bias = use_bias

    def compute_output_shape(self, input_shape):
        return self.layer.input.shape
    
    def get_config(self):
        config = super().get_config()
        layer_config = keras.saving.serialize_keras_object(self.layer)
        # self.constant is a tensor, first convert it to float value
        dico_params = {}
        dico_params["layer"] = layer_config
        dico_params["use_bias"] = self.use_bias
        # save input shape
        dico_params["input_shape"]= keras.saving.serialize_keras_object(self.layer.input.shape[1:])
        config.update(dico_params)
        return config

    @classmethod
    def from_config(cls, config):
        bias_config = config.pop("use_bias")
        use_bias = keras.saving.deserialize_keras_object(bias_config)
        layer_config = config.pop("layer")
        layer = keras.saving.deserialize_keras_object(layer_config)
        input_shape_config = config.pop("input_shape")
        input_shape = keras.saving.deserialize_keras_object(input_shape_config)

        # backward layers require the layer to have been built and have a predefined input/output shapes
        # this is why we store this information in the config file
        _ = keras.models.Sequential([layer])(np.ones(input_shape)[None])

        return cls(layer=layer, use_bias=use_bias, **config)
