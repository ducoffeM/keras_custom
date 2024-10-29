# define non native class Max
# Decomon Custom for Max(axis...)
import keras
import keras.ops as K
from keras.layers import Layer
import numpy as np

from typing import List


class Linear(keras.layers.Layer):
    """Custom Keras Layer that computes max on a Keras Tensor.
    """

    def __init__(self, layers:List[Layer], **kwargs):
        """
        Compute sequentially the list of linear layers
        """
        super(Linear, self).__init__(**kwargs)
        self.layers=layers

    def call(self, inputs_):
        output = inputs_
        for layer in self.layers:
            output = layer(output)

        return output

    def get_config(self):
        config = super().get_config()
        config_layers=[]
        for layer in self.layers:
            config_layers.append(keras.saving.serialize_keras_object(layer))
        
        config.update(
            {
                "layers":  config_layers,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        
        output_shape= input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)

        return output_shape

    @classmethod
    def from_config(cls, config):
        config_layers = config.pop('layers')
        layers = [keras.saving.deserialize_keras_object(config_layer) for config_layer in config_layers]
        return cls(layers=layers, **config)