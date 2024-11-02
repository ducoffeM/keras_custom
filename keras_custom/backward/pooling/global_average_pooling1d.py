import numpy as np
import keras
from keras.layers import Layer, GlobalAveragePooling1D
from keras_custom.backward.layer import BackwardLinearLayer
import keras.ops as K


# keepdims ????
class BackwardGlobalAveragePooling1D(BackwardLinearLayer):

    def call(self, inputs, training=None, mask=None):

        if self.layer.data_format == "channels_first":
            w_in = self.layer.input.shape[-1]
            if self.layer.keepdims:
                return K.repeat(inputs, w_in, -1) / w_in
            else:
                return K.repeat(K.expand_dims(inputs, -1), w_in, -1) / w_in
        else:
            w_in = self.layer.input.shape[1]
            if self.layer.keepdims:
                return K.repeat(inputs, w_in, 1) / w_in
            else:
                return K.repeat(K.expand_dims(inputs, 1), w_in, 1) / w_in


def get_backward_GlobalAveragePooling1D(layer: GlobalAveragePooling1D, use_bias=True) -> Layer:

    layer_backward = BackwardGlobalAveragePooling1D(layer)
    return layer_backward
