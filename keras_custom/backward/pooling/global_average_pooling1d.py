import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling1D
import keras.ops as K


class BackwardGlobalAveragePooling1D(Layer):

    def __init__(
        self,
        layer: GlobalAveragePooling1D,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer = layer
        if not self.layer.built:
            raise ValueError("layer {} is not built".format(layer.name))


    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    # serialize ...

    def call(self, inputs, training=None, mask=None):

        if self.layer.data_format == "channels_first":
            w_in = self.layer.input.shape[-1]
            return K.repeat(K.expand_dims(inputs, -1), w_in, -1)/w_in
        else:
            w_in = self.layer.input.shape[1]
            return K.repeat(K.expand_dims(inputs, 1), w_in, 1)/w_in


def get_backward_GlobalAveragePooling1D(layer: GlobalAveragePooling1D, use_bias=True) -> Layer:

    layer_backward = BackwardGlobalAveragePooling1D(layer)
    return layer_backward
