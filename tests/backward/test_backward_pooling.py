import keras
from keras.layers import AveragePooling2D, AveragePooling1D, AveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D
# comparison with depthwiseConv for AveragePooling
from keras.layers import DepthwiseConv2D, DepthwiseConv1D
from keras.models import Sequential
from keras_custom.backward import get_backward
import numpy as np
from .conftest import linear_mapping, serialize
import pytest

####### AveragePooling2D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding):

    # data_format == 'channels_first'
    layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_first')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    # equivalent DepthwiseConv2D
    pool_size = list(layer.pool_size)
    layer_conv = DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=layer.pool_size,
        strides=layer.strides,
        padding=layer.padding,
        data_format=layer.data_format,
        use_bias=False,
        trainable=False,
    )
    kernel_ = np.ones(pool_size + [1, 1]) / np.prod(pool_size)
    #layer_conv.weights = [keras.Variable(kernel_)]
    layer_conv(layer.input)
    layer_conv.built = True
    layer_conv.set_weights([kernel_])

    # check equality
    if padding=="valid":
        random_input = np.reshape(np.random.rand(np.prod(input_shape)*5), [5]+list(input_shape))
        output_pooling = layer(random_input)
        output_conv = layer_conv(random_input)
        np.testing.assert_almost_equal(output_pooling.cpu().numpy(), output_conv.cpu().numpy(), decimal=5)


    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)
    
    # data_format == 'channels_last'
    
    input_shape=input_shape[::-1]
    layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_last')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

    

def test_backward_AveragePooling2D():
    
    pool_size = (2, 2)
    strides=(1, 1)
    padding = 'valid'
    input_shape = (1,32, 32)
    _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)
    
    pool_size = (3, 3)
    strides=(2, 1)
    padding = 'valid'
    input_shape = (1,31, 32)
    _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)

    # not working: same NotImplementedError
    with pytest.raises(NotImplementedError):
        pool_size = (3, 3)
        strides=(2, 2)
        padding = 'same'
        input_shape = (1,31, 32)
        _test_backward_AveragePooling2D(input_shape, pool_size, strides, padding)

####### AveragePooling1D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_AveragePooling1D(input_shape, pool_size, strides, padding):

    # data_format == 'channels_first'
    layer = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_first')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    # equivalent DepthwiseConv1D
    pool_size = list(layer.pool_size)
    layer_conv = DepthwiseConv1D(
        depth_multiplier=1,
        kernel_size=layer.pool_size,
        strides=layer.strides,
        padding=layer.padding,
        data_format=layer.data_format,
        use_bias=False,
        trainable=False,
    )

    kernel_ = np.ones(pool_size + [1, 1]) / np.prod(pool_size)
    #layer_conv.weights = [keras.Variable(kernel_)]
    layer_conv(layer.input)
    layer_conv.built = True
    layer_conv.set_weights([kernel_])


    # check equality
    if padding=="valid":
        random_input = np.reshape(np.random.rand(np.prod(input_shape)*5), [5]+list(input_shape))
        output_pooling = layer(random_input)
        output_conv = layer_conv(random_input)
        np.testing.assert_almost_equal(output_pooling.cpu().numpy(), output_conv.cpu().numpy(), decimal=5)


    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


    # data_format == 'channels_last'
    input_shape=input_shape[::-1]
    layer = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_last')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

def test_backward_AveragePooling1D():
    
    pool_size = (2,)
    strides=(1,)
    padding = 'valid'
    input_shape = (1,32)
    _test_backward_AveragePooling1D(input_shape, pool_size, strides, padding)
    
    pool_size = (3,)
    strides=(2,)
    padding = 'valid'
    input_shape = (1,31)
    _test_backward_AveragePooling1D(input_shape, pool_size, strides, padding)

    # not working: same NotImplementedError
    with pytest.raises(NotImplementedError):
        pool_size = (3,)
        strides=(2,)
        padding = 'same'
        input_shape = (1,31)
        _test_backward_AveragePooling1D(input_shape, pool_size, strides, padding)

####### AveragePooling3D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_AveragePooling3D(input_shape, pool_size, strides, padding):

    # data_format == 'channels_first'
    layer = AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_first')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)


    # data_format == 'channels_last'
    input_shape=input_shape[::-1]
    layer = AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format = 'channels_last')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

def test_backward_AveragePooling3D():
    
    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend()=='torch':
        import torch
        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as Conv3DTranspose is not implemented")

    pool_size = (2,2, 2)
    strides=(1,1, 1)
    padding = 'valid'
    input_shape = (1,32, 32, 32)
    _test_backward_AveragePooling3D(input_shape, pool_size, strides, padding)
    
    pool_size = (3,1, 4)
    strides=(2,1, 1)
    padding = 'valid'
    input_shape = (1,31, 30, 32)
    _test_backward_AveragePooling3D(input_shape, pool_size, strides, padding)

    # not working: same NotImplementedError
    with pytest.raises(NotImplementedError):
        pool_size = (3,1, 4)
        strides=(2,1, 1)
        padding = 'same'
        input_shape = (1,31, 30, 32)
        _test_backward_AveragePooling3D(input_shape, pool_size, strides, padding)


####### GlobalAveragePooling2D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_GlobalAveragePooling2D(input_shape,keepdims):

    # data_format == 'channels_first'
    layer = GlobalAveragePooling2D(keepdims=keepdims, data_format = 'channels_first')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)
    
    # data_format == 'channels_last'
    
    input_shape=input_shape[::-1]
    layer = GlobalAveragePooling2D(keepdims=keepdims, data_format = 'channels_last')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

def test_backward_GlobalAveragePooling2D():
    

    input_shape = (3,32, 32)
    _test_backward_GlobalAveragePooling2D(input_shape, keepdims=True)
    _test_backward_GlobalAveragePooling2D(input_shape, keepdims=False)

    input_shape = (2,31, 32)
    _test_backward_GlobalAveragePooling2D(input_shape, keepdims=True)
    _test_backward_GlobalAveragePooling2D(input_shape, keepdims=False)

####### GlobalAveragePooling1D #######
# pool_size, strides=None, padding="valid", data_format=None
def _test_backward_GlobalAveragePooling1D(input_shape,keepdims):

    # data_format == 'channels_first'
    layer = GlobalAveragePooling1D(keepdims=keepdims, data_format = 'channels_first')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])

    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)
    
    # data_format == 'channels_last'
    
    input_shape=input_shape[::-1]
    layer = GlobalAveragePooling1D(keepdims=keepdims, data_format = 'channels_last')
    model_layer = Sequential([layer])
    _ = model_layer(np.ones(input_shape)[None])
    backward_layer = get_backward(layer, use_bias=False)
    linear_mapping(layer, backward_layer)
    # use_bias should have an impact
    backward_layer = get_backward(layer, use_bias=True)
    serialize(layer, backward_layer)

def test_backward_GlobalAveragePooling1D():
    
    input_shape = (3,32)
    _test_backward_GlobalAveragePooling1D(input_shape, keepdims=True)
    _test_backward_GlobalAveragePooling1D(input_shape, keepdims=False)

    input_shape = (2,31)
    _test_backward_GlobalAveragePooling1D(input_shape, keepdims=True)
    _test_backward_GlobalAveragePooling1D(input_shape, keepdims=False)



    



