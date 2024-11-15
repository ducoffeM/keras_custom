from keras.layers import Dense, Reshape, Flatten, ReLU, Conv2D, DepthwiseConv2D
from keras.models import Sequential
from keras_custom.backward.models import get_backward_sequential
from .conftest import compute_backward_model, serialize_model
import numpy as np
import torch
"""
def test_sequential_linear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is linear
    _ = backward_model(np.ones((1,1)))
    compute_backward_model((input_dim,), model, backward_model)
    serialize_model([32], model)
    serialize_model([1], backward_model)
"""

def test_sequential_nonlinear():
    input_dim = 32
    layers = [Dense(2, use_bias=False), ReLU(), Dense(1, use_bias=False)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([np.ones((1, input_dim)), np.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    #serialize_model([input_dim], model)
    serialize_model([input_dim, 1], backward_model)

def test_sequential_multiD():

    input_dim = 36
    layers = [Reshape((1, 6, 6)), Conv2D(2, (3, 3)), ReLU(), Reshape((-1,)), Dense(1)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    #serialize_model([input_dim, 1], backward_model)

def test_sequential_multiD_channel_last():

    input_dim = 72
    layers = [Reshape((6, 6, 2)), DepthwiseConv2D(2, (3, 3), data_format="channels_last"), ReLU(), Reshape((-1,)), Dense(1)]
    model = Sequential(layers)
    _ = model(torch.ones((1, input_dim)))
    backward_model = get_backward_sequential(model)
    # model is not linear
    _ = backward_model([torch.ones((1, input_dim)), torch.ones((1,1))])
    compute_backward_model((input_dim,), model, backward_model)
    #serialize_model([input_dim, 1], backward_model)




