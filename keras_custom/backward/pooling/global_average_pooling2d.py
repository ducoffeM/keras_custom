import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling2D
import keras.ops as K


class BackwardGlobalAveragePooling2D(Layer):

    def __init__(
        self,
        layer: GlobalAveragePooling2D,
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
            w_in, h_in = self.layer.input.shape[-2:]
            if self.layer.keepdims:
                return K.repeat(K.repeat(inputs, w_in, -2), h_in, -1) / (w_in * h_in)
            else:
                return K.repeat(K.repeat(K.expand_dims(K.expand_dims(inputs, -1), -1), w_in, -2), h_in, -1) / (
                    w_in * h_in
                )
        else:
            w_in, h_in = self.layer.input.shape[1:3]
            if self.layer.keepdims:
                return K.repeat(K.repeat(inputs, w_in, 1), h_in, 2) / (w_in * h_in)
            else:
                return K.repeat(K.repeat(K.expand_dims(K.expand_dims(inputs, 1), 1), w_in, 1), h_in, 2) / (w_in * h_in)


def get_backward_GlobalAveragePooling2D(layer: GlobalAveragePooling2D, use_bias=True) -> Layer:

    layer_backward = BackwardGlobalAveragePooling2D(layer)
    return layer_backward
