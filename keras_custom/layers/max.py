# define non native class Max
# Decomon Custom for Max(axis...)
import keras
import keras.ops as K
import numpy as np

from typing import Tuple, List


class Max(keras.layers.Layer):
    """Custom Keras Layer that computes max on a Keras Tensor.
    """

    def __init__(self, axis:int, keepdims=True, **kwargs):
        """
        Compute the result of  max(x) along a given axis: K.max(x, axis)
        Args:
            axis: The axis dimension on which we perform the operator.
        """
        super(Max, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims=keepdims

    def call(self, inputs_):
        return keras.ops.max(inputs_, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis":  keras.saving.serialize_keras_object(self.axis),
                "keepdims": keras.saving.serialize_keras_object(self.keepdims),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        
        input_shape = list(input_shape)
        if self.axis<0:
            axis_ = len(input_shape)+self.axis
        else:
            axis_ = self.axis

        if self.keepdims:
            tmp_shape=[1]
        else:
            tmp_shape=[]
        return input_shape[:axis_]+tmp_shape+input_shape[axis_+1:]

    @classmethod
    def from_config(cls, config):
        axis_config = config.pop('axis')
        keepdims_config = config.pop('keepdims')
        axis = keras.saving.deserialize_keras_object(axis_config)
        keepdims = keras.saving.deserialize_keras_object(keepdims_config)
        return cls(axis=axis, keepdims=keepdims, **config)