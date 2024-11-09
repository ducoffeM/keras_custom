from keras.layers import Dense
from keras.models import Sequential
from keras_custom.backward import get_backward
import numpy as np
from .conftest import linear_mapping, serialize


def test_backward_Dense():
    
    layer = Dense(units=3, use_bias=False)
    model_layer = Sequential([layer])
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    
    layer = Dense(units=3, use_bias=False)
    model_layer = Sequential([layer])
    input_shape = (2,32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    serialize(layer, backward_layer)
    """
    layer_0 = Dense(units=3, use_bias=False)
    model_layer_0 = Sequential([layer_0])
    input_shape = (2,32, 31)
    _ = model_layer_0(np.ones(input_shape)[None])
    backward_layer_0 = get_backward(layer_0, use_bias=False)
    serialize(layer_0, backward_layer_0)
    linear_mapping(layer_0, backward_layer_0)
    serialize(layer_0, backward_layer_0)
    """