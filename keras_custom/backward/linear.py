from keras_custom.backward.layers.layer import BackwardLinearLayer
from keras_custom.backward.layers.convert import get_backward as get_backward_
from keras_custom.layers import Linear
from keras.layers import Layer


def convert_to_backward(layer, use_bias):
    if isinstance(layer, BackwardLinearLayer):
        layer_backward = BackwardLinear(layer, use_bias)
    else:
        layer_backward = get_backward_(layer, use_bias)
    return layer_backward

class BackwardLinear(BackwardLinearLayer):
    def __init__(
        self,
        layer: Linear,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
        layers_backward = [convert_to_backward(layer, self.use_bias) for layer in self.layer.layers][::-1]
        self.layer_backward = Linear(layers_backward)

    def call(self, inputs, training=None, mask=None):
        return self.layer_backward(inputs)

def get_backward_Linear(layer: Linear, use_bias=True) -> Layer:
    return BackwardLinear(layer, use_bias)