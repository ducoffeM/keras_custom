from keras.layers import Flatten, Reshape
from keras.layers import Layer

def get_backward_Flatten(layer: Flatten, use_bias=True) -> Layer:

    input_shape_wo_batch = layer.input.shape[1:]
    return Reshape(target_shape = input_shape_wo_batch)