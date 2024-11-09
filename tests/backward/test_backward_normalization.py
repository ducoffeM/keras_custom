from keras.layers import BatchNormalization
from keras.models import Sequential
from keras_custom.backward import get_backward
import numpy as np
from .conftest import linear_mapping, is_invertible, serialize


def test_backward_BatchNormalization():

    layer = BatchNormalization()
    layer.trainable = False
    model_layer = Sequential([layer])
    model_layer.trainable = False
    input_shape = (2,)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)

    """
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable=False
    model_layer = Sequential([layer])
    model_layer.trainable=False
    input_shape = (1,32)
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    layer = BatchNormalization()
    layer.trainable=False
    model_layer = Sequential([layer])
    model_layer.trainable=False
    input_shape = (1,32,32)
    _ = model_layer(np.ones(input_shape)[None])
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    is_invertible(layer, backward_layer)
    serialize(layer, backward_layer)

    # do some training
    """
