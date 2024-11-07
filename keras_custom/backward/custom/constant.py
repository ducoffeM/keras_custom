from keras_custom.layers import MulConstant, PlusConstant
from keras.layers import Layer
from keras_custom.backward.layer import BackwardLinearLayer


class BackwardMulConstant(BackwardLinearLayer):
    def __init__(
        self,
        layer: MulConstant,
        **kwargs,
    ):
        super().__init__(layer=layer, **kwargs)
        self.layer_backward = MulConstant(constant=1.0 / layer.constant)
        self.layer_backward.built = True

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_MulConstant(layer: MulConstant, use_bias=True) -> Layer:

    return BackwardMulConstant(layer)


class BackwardPlusConstant(BackwardLinearLayer):
    def __init__(
        self,
        layer: PlusConstant,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
        if self.use_bias:
            self.layer_backward = PlusConstant(constant=-layer.constant * layer.sign, minus=(layer.sign == -1))
        else:
            self.layer_backward = PlusConstant(constant=0.0, minus=(layer.sign == -1))

        self.layer_backward.built = True

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)


def get_backward_PlusConstant(layer: PlusConstant, use_bias=True) -> Layer:
    return BackwardPlusConstant(layer, use_bias)
