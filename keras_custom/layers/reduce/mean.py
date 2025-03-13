import keras
from keras_custom.layers.core import BaseAxisKeepdimsLayer


class Mean(BaseAxisKeepdimsLayer):
    """
    Custom Keras Layer that computes the mean of elements along a specified axis.
    Inherits axis and keepdims attributes from BaseAxisKeepdimsLayer.
    """

    def call(self, inputs_):
        """Computes the mean along the specified axis, retaining dimensions if keepdims is True."""
        return keras.ops.mean(inputs_, keepdims=self.keepdims, axis=self.axis)
