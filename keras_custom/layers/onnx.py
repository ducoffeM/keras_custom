import keras
from typing import Union
import numpy as np

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