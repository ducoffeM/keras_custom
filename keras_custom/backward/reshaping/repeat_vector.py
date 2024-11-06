from keras.layers import RepeatVector, Layer


class BackwardRepeatVector(Layer):

    def __init__(
        self,
        layer: RepeatVector,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer = layer

    def call(self, inputs, training=None, mask=None):

        return inputs[:,0]


def get_backward_RepeatVector(layer: BackwardRepeatVector, use_bias=True) -> Layer:

    layer_backward = BackwardRepeatVector(layer)
    return layer_backward
