import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model
from keras_custom.numpy import (
    Abs,
    Absolute,
    AMax,
    AMin,
    Append,
    Arccos,
    Arccosh,
    Arcsin,
    Arcsinh,
    Arctan,
    Arctan2,
    Arctanh,
    Average,
    Cos,
    Cosh,
    Cross,
    Cumprod,
    Cumsum,
    Diag,
    Diagonal,
    ExpandDims,
    Expm1,
    Flip,
    Floor,
    FullLike,
    GetItem,
    Hstack,
    Log,
    Log1p,
    Log2,
    Log10,
    LogAddExp,
    Maximum,
    Minimum,
    MoveAxis,
    Negative,
    OnesLike,
    Reciprocal,
    Repeat,
    ZerosLike,
)

from .conftest import func_layer, func_layer_binary


def _test_ops_unary(keras_layer, input_shape):
    # designed for layers taking a single input
    batch_size = 10
    keras_ops = keras_layer.get_ops()
    func_layer(keras_layer, input_shape)

    x = Input(input_shape)
    model_layer = Model(x, keras_layer(x))
    model_ops = Model(x, keras_ops(x))

    toy_value_np = np.reshape(
        np.random.rand(batch_size, np.prod(input_shape)), (batch_size,) + input_shape
    )

    output_layer = model_layer.predict(toy_value_np, verbose=0)
    output_ops = model_ops.predict(toy_value_np, verbose=0)
    np.testing.assert_almost_equal(output_layer, output_ops, decimal=5)


def _test_ops_binary(keras_layer, input_shape):
    # designed for layers taking a single input
    batch_size = 10
    keras_ops = keras_layer.get_ops()
    func_layer_binary(keras_layer, input_shape)

    x = Input(input_shape)
    y = Input(input_shape)
    model_layer = Model([x, y], keras_layer([x, y]))
    model_ops = Model([x, y], keras_ops([x, y]))

    toy_value_np_x = np.reshape(
        np.random.rand(batch_size, np.prod(input_shape)), (batch_size,) + input_shape
    )
    toy_value_np_y = np.reshape(
        np.random.rand(batch_size, np.prod(input_shape)), (batch_size,) + input_shape
    )

    output_layer = model_layer.predict([toy_value_np_x, toy_value_np_y], verbose=0)
    output_ops = model_ops.predict([toy_value_np_x, toy_value_np_y], verbose=0)
    np.testing.assert_almost_equal(output_layer, output_ops, decimal=5)


@pytest.mark.parametrize(
    "keras_layer",
    [
        Abs(),
        Absolute(),
        AMax(axis=-1, keepdims=True),
        AMin(axis=-1, keepdims=True),
        Arccos(),
        Arccosh(),
        Arcsin(),
        Arcsinh(),
        Arctan(),
        Arctanh(),
        Average(axis=-1),
        Cos(),
        Cosh(),
        Cumprod(axis=-1),
        Cumsum(axis=-1),
        Diagonal(axis1=1, axis2=2),
        ExpandDims(axis=-1),
        Expm1(),
        Floor(),
        FullLike(3),
        Log(),
        Log10(),
        Log1p(),
        Log2(),
        MoveAxis(2, 1),
        Negative(),
        OnesLike(),
        Reciprocal(),
        Repeat(repeats=3, axis=-1),
        ZerosLike,
    ],
)
def test_unary_ops(keras_layer):
    input_shape = (2, 3)
    _test_ops_unary(keras_layer, input_shape)


@pytest.mark.parametrize(
    "keras_layer",
    [Diag(k=1), Diagonal(), Flip(axis=-1), GetItem(1)],
)
def test_unary_vector_ops(keras_layer):
    input_shape = (4,)  # for 2D tensor including batch size
    _test_ops_unary(keras_layer, input_shape)


@pytest.mark.parametrize(
    "keras_layer",
    [Append(axis=-1), Arctan2(), Average(axis=-1), Cross(), LogAddExp(), Maximum(), Minimum()],
)
def test_binary_ops(keras_layer):
    input_shape = (2, 3)
    _test_ops_binary(keras_layer, input_shape)


@pytest.mark.parametrize(
    "keras_layer",
    [
        Hstack(),
    ],
)
def test_binary_vector_ops(keras_layer):
    input_shape = (4,)
    _test_ops_binary(keras_layer, input_shape)
