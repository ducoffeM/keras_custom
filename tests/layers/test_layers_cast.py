from .conftest import func_layer
from keras_custom.layers import Cast

def test_Cast():

    layer = Cast(1)

    input_shape = (2,)
    func_layer(layer, input_shape)

    layer = Cast(2)
    input_shape = (1, 32)
    func_layer(layer, input_shape)

    layer = Cast(3)
    input_shape = (1, 32, 32)
    func_layer(layer, input_shape)