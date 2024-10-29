import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling3D
import keras.ops as K


class BackwardGlobalAveragePooling3D(Layer):

    def __init__(
        self,
        layer: GlobalAveragePooling3D,
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
            d_in, w_in, h_in = self.layer.input.shape[-3:]
            return K.repeat(
                K.repeat(K.repeat(K.expand_dims(K.expand_dims(K.expand_dims(inputs, -1), -1), -1), d_in, -3), w_in, -2),
                h_in,
                -1,
            ) / (w_in * h_in * d_in)
        else:
            d_in, w_in, h_in = self.layer.input.shape[1:4]
            return K.repeat(
                K.repeat(K.repeat(K.expand_dims(K.expand_dims(K.expand_dims(inputs, 1), 1), 1), d_in, 1), w_in, 2),
                h_in,
                3,
            ) / (w_in * h_in * d_in)


def get_backward_GlobalAveragePooling3D(layer: GlobalAveragePooling3D, use_bias=True) -> Layer:

    layer_backward = BackwardGlobalAveragePooling3D(layer)
    return layer_backward
