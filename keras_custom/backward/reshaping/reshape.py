from keras.layers import Reshape
from keras.layers import Layer

def get_backward_Reshape(layer: Reshape, use_bias=True) -> Layer:

    input_shape_wo_batch = layer.input.shape[1:]
    return Reshape(target_shape = input_shape_wo_batch)