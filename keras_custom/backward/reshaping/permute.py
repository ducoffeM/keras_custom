from keras.layers import Permute
from keras.layers import Layer

def get_backward_Permute(layer: Permute, use_bias=True) -> Layer:

    return layer