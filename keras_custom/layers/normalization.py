import keras
from keras_custom.layers.core import BaseAxisKeepdimsLayer


class ReduceL2(BaseAxisKeepdimsLayer):
    """
    Custom Keras Layer that computes the L2 norm (Euclidean norm) along a specified axis.
    Inherits axis and keepdims attributes from BaseAxisKeepdimsLayer.
    """

    def call(self, inputs_):
        """Computes the L2 norm along the specified axis, retaining dimensions if keepdims is True."""
        return keras.ops.norm(inputs_, axis=self.axis, keepdims=self.keepdims)
