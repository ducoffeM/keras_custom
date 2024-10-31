from .conftest import func_layer
from keras_custom.layers import Sum


def test_Sum():

    layer = Sum(axis=-1, keepdims=True)

    input_shape = (2,)
    func_layer(layer, input_shape)

    layer = Sum(axis=1, keepdims=True)
    input_shape = (1, 32)
    func_layer(layer, input_shape)

    layer = Sum(axis=2, keepdims=False)
    input_shape = (1, 32, 32)
    func_layer(layer, input_shape)