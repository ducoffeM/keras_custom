import keras
from typing import Union
import numpy as np

import numpy
ArrayLike = numpy.typing.ArrayLike  # TO DO typing repo


# custom layers used during onnx export: onnx2keras3


class PlusConstant(keras.layers.Layer):
    """Custom Keras Layer that adds a constant value to a Keras Tensor.
    This layer performs element-wise addition of a constant value to a Keras Tensor.
    """

    def __init__(self, constant, minus = False, **kwargs):
        """
        Compute the result of (-1 * x + constant) or (x + constant), depending on the 'minus' parameter.
        Args:
            constant: The constant value to be added to the tensor.
            minus: The indicator for the operation to be performed:
                 - If minus equals 1, it computes (-1 * x + constant).
                 - If minus equals -1, it computes (x + constant).
        """
        super(PlusConstant, self).__init__(**kwargs)
        self.constant = keras.Variable(constant)
        self.sign: int = 1
        if minus:
            self.sign = -1

    def call(self, inputs_):
        return self.sign * inputs_ + self.constant

    def get_config(self):
        config = super().get_config()
        # self.constant is a tensor, first convert it to float value
        const_ = self.constant.numpy()
        dico_params={}
        dico_params['constant']=const_
        dico_params['sign']=self.sign
        config.update(dico_params)
        #config.update({"constant": const_, "sign": self.sign})
        return config

    def compute_output_shape(self, input_shape):

        return input_shape

    @classmethod
    def from_config(cls, config):
        constant_config = config.pop('constant')
        constant = keras.saving.deserialize_keras_object(constant_config)
        sign_config = config.pop('sign')
        sign = keras.saving.deserialize_keras_object(sign_config)
        if sign>0:
            minus = False
        else:
            minus = True
        return cls(constant=constant, minus=minus, **config) 


class MulConstant(keras.layers.Layer):
    """Custom Keras Layer that multiply a constant value to a Keras Tensor.
    This layer performs element-wise multiplication of a constant value to a Keras Tensor.
    """

    def __init__(self, constant, **kwargs):
        """
        Compute the result of  x*constant.
        Args:
            constant: The constant value to be elementwise multiplied with the tensor.
        """
        super(MulConstant, self).__init__(**kwargs)
        if not isinstance(constant, float) and len(constant.shape):
            self.constant = keras.ops.convert_to_tensor(constant)
        else:
            self.constant = constant

    def call(self, inputs_):
        return keras.ops.multiply(inputs_, self.constant)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "constant":  keras.saving.serialize_keras_object(self.constant),
            }
        )
        return config

    def compute_output_shape(self, input_shape):

        return input_shape

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop('constant')
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config) 


class Pow(keras.layers.Layer):

    def __init__(self, power: int, **kwargs):
        super(Pow, self).__init__(**kwargs)
        self.power: int = power

    def call(self, inputs_):
        return keras.ops.power(inputs_, self.power)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"power": self.power})
        return config


class Split(keras.layers.Layer):
    def __init__(self, splits, axis, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.splits = list(splits)
        self.axis = axis

    def call(self, inputs_):
        output = keras.ops.split(inputs_, indices_or_sections=self.splits, axis=self.axis)
        output_shape = [e.shape for e in output]
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "splits": self.splits,
                # "i": self.i,
                "axis": self.axis,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        input_shape_ = list(input_shape)
        input_shape_ = input_shape_[1:]
        tmp = np.ones(input_shape_)[None]

        output = np.split(tmp, indices_or_sections=self.splits, axis=self.axis)
        return [e.shape for e in output]
    
class DivConstant(keras.layers.Layer):
    """Custom Keras Layer that divide a constant value with a Keras Tensor.
    This layer performs element-wise division of a constant value and a Keras Tensor.
    """

    def __init__(self, constant: Union[float, ArrayLike], **kwargs):
        """
        Compute the result of  x*constant.
        Args:
            constant: The constant value to be elementwise multiplied with the tensor.
        """
        super(DivConstant, self).__init__(**kwargs)

        if len(constant.shape):
            self.constant: Union[float, Tensor] = keras.ops.convert_to_tensor(constant)
        else:
            self.constant: Union[float, Tensor] = constant

    def call(self, inputs_):
        return self.constant / inputs_

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "constant": self.constant,
            }
        )
        return config

    def compute_output_shape(self, input_shape):

        return input_shape
    

from typing import List, Dict, Any, Union

import keras
import keras.backend as K
import numpy as np
from keras.layers import Layer, Lambda


class Clip(keras.layers.Layer):
    """Custom Keras Layer that clip a Keras Tensor with two constant values vmin, vmax.
    This layer performs element-wise clipping.
    """

    def __init__(self, vim: float, vmax: float, **kwargs):
        """
        Compute clipping
        Args:
            vmin: elementwise lower bound
            vmax: elementwise upper bound
        """
        super(Clip, self).__init__(**kwargs)
        self.vmin: float = vim
        self.vmax: float = vmax

    def call(self, inputs_):

        return keras.ops.clip(inputs_, vmin, vmax)

    def get_config(self):
        config = super().get_config()
        config.update({"vmin": self.vmin, "vmax": self.vmax})
        return config

    def compute_output_spec(self, *args, **kwargs):

        return args[0]


class Log(keras.layers.Layer):
    """Custom Keras Layer that compute log on a Keras Tensor.
    This layer performs element-wise log.
    """

    def call(self, inputs_):

        return keras.ops.log(inputs_)

    def compute_output_spec(self, *args, **kwargs):

        return args[0]


class Sum(Layer):
    """This custom Keras Layer computes the sum of elements along a specified axis of a Keras Tensor."""

    def __init__(self, axis=None, **kwargs):
        super(Sum, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs_):
        return keras.ops.sum(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config

    def compute_output_spec(self, *args, **kwargs):
        if self.axis is None:
            return keras.ops.sum(args[0], axis=self.axis, keepdims=self.keepdims)
        else:
            return keras.ops.split(args[0], [1], self.axis)[0]


class Mean(Layer):
    """This custom Keras Layer computes the mean of elements along a specified axis of a Keras Tensor."""

    def __init__(self, axis=None, keepdims=True, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs_):
        return keras.ops.mean(inputs_, keepdims=self.keepdims, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config

    def compute_output_spec(self, *args, **kwargs):

        if self.axis is None:
            return keras.ops.mean(args[0], axis=self.axis, keepdims=self.keepdims)
        else:
            return keras.ops.split(args[0], [1], self.axis)[0]


class Max(Layer):
    """This custom Keras Layer computes the max of elements along a specified axis of a Keras Tensor."""

    def __init__(self, axis=None, keepdims=True, **kwargs):
        super(Max, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs_):
        return keras.ops.mean(inputs_, keepdims=self.keepdims, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config

    def compute_output_spec(self, *args, **kwargs):

        if self.axis is None:
            return keras.ops.mean(args[0], axis=self.axis, keepdims=self.keepdims)
        else:
            return keras.ops.split(args[0], [1], self.axis)[0]


class Split(Layer):
    def __init__(self, splits, axis, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.splits = list(splits)
        # self.i = i
        self.axis = axis

    def call(self, inputs_):
        return keras.ops.split(inputs_, indices_or_sections=self.splits, axis=self.axis)
        # return keras.ops.split(inputs_, num_or_size_splits=self.splits, axis=self.axis)#[self.i]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "splits": self.splits,
                # "i": self.i,
                "axis": self.axis,
            }
        )
        return config

    def compute_output_spec(self, *args, **kwargs):

        return keras.ops.split(args[0], indices_or_sections=self.splits, axis=self.axis)


class Identity(Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs_):
        return inputs_

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):

        return input_shape


class Cast(Layer):
    def __init__(self, dtype: int, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.dtype: int = dtype
        self.cast_map = {
            1: K.float32,
            2: K.uint8,
            3: K.int8,
            5: K.int16,
            6: K.int32,
            7: K.int64,
            9: K.bool,
            10: K.float16,
            11: K.double,
        }

    def call(self, inputs_):
        return keras.ops.cast(inputs_, self.cast_map[self.dtype])

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self.dtype, "cast_map": self.cast_map})
        return config

    def compute_output_shape(self, input_shape):

        return input_shape


class Floor(Layer):

    def call(self, inputs_):
        return keras.ops.floor(inputs_)

    def compute_output_shape(self, input_shape):

        return input_shape


class Argmax(Layer):
    def __init__(self, axis: Union[None, int], **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.axis: int = dtype

    def call(self, inputs_):
        return keras.ops.argmax(inputs_, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_spec(self, *args, **kwargs):

        return keras.ops.sum(args[0], self.axis)


class ReduceL2(Layer):
    def __init__(self, axis: int, keepdims: bool, **kwargs):
        super(ReduceL2, self).__init__(**kwargs)
        self.axis: int = axis
        self.keepdims = keepdims

    def call(self, inputs_):
        return keras.ops.norm(inputs_, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config

    def compute_output_spec(self, *args, **kwargs):

        return keras.ops.sum(args[0], axis=self.axis, keepdims=self.keepdims)


class Slice(Layer):

    def __init__(self, axis, starts, ends, steps, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.axis = axis
        self.starts = starts
        self.ends = ends
        self.steps = steps

        if self.axis[0] not in [2, 3]:
            raise ValueError(axis[0])

    def call(self, inputs_):
        if axes[0] == 2:
            return inputs_[:, :, self.starts[0] : self.ends[0]][:, :, :: self.steps[0]]
        elif axes[0] == 3:
            return inputs_[:, :, :, starts[0] : ends[0]][:, :, :, :: steps[0]]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "starts": self.starts, "ends": self.ends, "steps": self.steps})
        return config

    def compute_output_spec(self, *args, **kwargs):
        if self.axes[0] == 2:
            return args[0][:, :, self.starts[0] : self.ends[0]][:, :, :: self.steps[0]]
        elif self.axes[0] == 3:
            return args[0][:, :, :, starts[0] : ends[0]][:, :, :, :: steps[0]]
