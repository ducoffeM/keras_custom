from keras.layers import Dense, ReLU, Input
from keras.models import Sequential, Model
from keras_custom.backward import get_backward
import numpy as np
from .conftest import compute_backward, serialize
import keras.ops as K


def _test_backward_activation(layer):

    input_shape = (32,)
    model = Sequential([layer])
    model(K.ones([1]+list(input_shape)))

    backward_layer = get_backward(layer)

    mask_output = Input(input_shape)
    input_ = Input(input_shape)
    output = backward_layer([mask_output, input_])
    model_backward = Model([mask_output, input_], output)

    compute_backward(input_shape, model, model_backward)



def test_backward_ReLU():
    layer = ReLU(threshold=1., negative_slope=0.3)
    _test_backward_activation(layer)

    #layer = ReLU(threshold=1.1, negative_slope=0.3)
    #_test_backward_activation(layer)