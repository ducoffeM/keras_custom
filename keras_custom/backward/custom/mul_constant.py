from keras_custom.layers.onnx import MulConstant
from keras.layers import Layer

def get_backward_MulConstant(layer: MulConstant, use_bias=True) -> Layer:

    layer_backward = MulConstant(constant=1./layer.constant)
    layer_backward.built = True

    return layer_backward