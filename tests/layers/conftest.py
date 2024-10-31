import os
import numpy as np
import keras
from keras_custom.layers import *
from keras.layers import Dense, Reshape, ReLU
from keras.models import Sequential
import torch

# test custom layers
# to test: initialization, inference, serialization and deserialization


def func_layer(layer, input_shape):

    # create a toy model: Dense(np.prod(input_shape)), Reshape(input_shape), layer, Reshape(-1), Dense(1)
    dense_0 = Dense(np.prod(input_shape))
    reshape_0 = Reshape(input_shape)
    reshape_1 = Reshape((-1,))
    dense_1 = Dense(1)
    if not isinstance(layer, list):
        layer = [layer]
    toy_model = Sequential(layers=[dense_0, reshape_0] + layer + [reshape_1, dense_1])

    toy_dataset = torch.ones((10, 2))
    toy_label = torch.ones((10, 1))
    _ = toy_model(toy_dataset)

    toy_model.compile("sgd", "mse")

    toy_model.fit(toy_dataset, toy_label, epochs=1, batch_size=2, verbose=0)

    toy_value_np = np.random.rand(1, 2)

    output_after_training = toy_model.predict(toy_value_np)
    # serialize
    filename = "test_serialize_{}_{}.keras".format(layer[0].__class__.__name__, layer[0].name)

    # detach toy model to cpu
    # toy_model.to('cpu')
    toy_model.save(filename)  # The file needs to end with the .keras extension

    # deserialize
    load_model = keras.models.load_model(filename)

    # compare with the previous output
    output_after_export = load_model.predict(toy_value_np)

    np.testing.assert_almost_equal(output_after_training, output_after_export, err_msg="corrupted weights")
    os.remove(filename)
