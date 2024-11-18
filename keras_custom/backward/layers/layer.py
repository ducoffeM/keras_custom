# abstract class for BackwardLayer
from keras.layers import Layer
import keras
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
import keras.ops as K
from keras_custom.backward.layers.utils import reshape_to_batch


class BackwardLayer(Layer):
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
        input_dim_wo_batch: Union[None, List[int]] = None,
        output_dim_wo_batch: Union[None, List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.use_bias = use_bias
        if input_dim_wo_batch is None:
            # can be a list ...
            input_dim_wo_batch = list(self.layer.input.shape[1:])

        self.input_dim_wo_batch = input_dim_wo_batch
        if isinstance(self.input_dim_wo_batch[0], list):
            self.n_input = len(input_dim_wo_batch)
        else:
            self.n_input = 1
        if output_dim_wo_batch is None:
            output_dim_wo_batch = list(self.layer.output.shape[1:])
        self.output_dim_wo_batch = output_dim_wo_batch

        # In many scenarios, the backward pass can be represented using a native Keras layer
        # (e.g., Conv2D can map to Conv2DTranspose).
        # In such cases, users can directly specify a `layer_backward` function,
        # which will be invoked automatically.
        self.layer_backward = None

    def get_config(self):
        config = super().get_config()
        layer_config = keras.saving.serialize_keras_object(self.layer)
        # self.constant is a tensor, first convert it to float value
        dico_params = {}
        dico_params["layer"] = layer_config
        dico_params["use_bias"] = self.use_bias
        # save input shape
        dico_params["input_dim_wo_batch"] = keras.saving.serialize_keras_object(self.input_dim_wo_batch)
        dico_params["output_dim_wo_batch"] = keras.saving.serialize_keras_object(self.output_dim_wo_batch)

        config.update(dico_params)

        return config

    def call_on_reshaped_gradient(self, gradient, input=None, training=None, mask=None):
        if self.layer_backward:
            return self.layer_backward(gradient)
        raise NotImplementedError()

    def call(self, inputs, training=None, mask=None):
        layer_input = None

        if not isinstance(inputs, list):
            gradient = inputs
        else:
            gradient = inputs[0]
            if len(inputs) == 2:
                layer_input = inputs[1]
            elif len(inputs) > 2:
                layer_input = inputs[1:]

        reshape_tag, gradient, n_out = reshape_to_batch(gradient, [1] + self.output_dim_wo_batch)
        
        output = self.call_on_reshaped_gradient(gradient, input=layer_input, training=training,mask=mask)

        if reshape_tag:
            if isinstance(output, list):
                output = [
                    K.reshape(output[i], [-1] + n_out + list(self.input_dim_wo_batch[i])) for i in range(self.n_input)
                ]
            else:
                output = K.reshape(output, [-1] + n_out + self.input_dim_wo_batch)

        return output

    @classmethod
    def from_config(cls, config):
        bias_config = config.pop("use_bias")
        use_bias = keras.saving.deserialize_keras_object(bias_config)
        layer_config = config.pop("layer")
        layer = keras.saving.deserialize_keras_object(layer_config)

        input_dim_wo_batch_config = config.pop("input_dim_wo_batch")
        input_dim_wo_batch = keras.saving.deserialize_keras_object(input_dim_wo_batch_config)
        output_dim_wo_batch_config = config.pop("output_dim_wo_batch")
        output_dim_wo_batch = keras.saving.deserialize_keras_object(output_dim_wo_batch_config)

        return cls(
            layer=layer,
            use_bias=use_bias,
            input_dim_wo_batch=input_dim_wo_batch,
            output_dim_wo_batch=output_dim_wo_batch,
            **config,
        )

    def compute_output_shape(self, input_shape):
        if isinstance(self.input_dim_wo_batch[0], list):
            return [[1]+input_dim_wo_batch_i for input_dim_wo_batch_i in self.input_dim_wo_batch]
        else:
            return [1] + self.input_dim_wo_batch


class BackwardLinearLayer(BackwardLayer):
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


class BackwardNonLinearLayer(BackwardLayer):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardNonLinearLayer` is designed to perform the reverse operation of a wrapped Keras
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
