#from keras_custom.backward.layers.layer import BackwardNonLinear
from keras.layers import Layer, ReLU
import keras.ops as K
from keras_custom.backward.layers.layer import BackwardNonLinearLayer


class BackwardReLU(BackwardNonLinearLayer):

    def __init__(
        self,
        layer: ReLU,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(layer=layer, use_bias=use_bias, **kwargs)
    
    def call(self, inputs, training=None, mask=None):
        layer_output, layer_input = inputs
        # compute derivative of relu
        mask_neg_slope = K.relu(K.sign(self.layer.threshold - layer_input)) # 1 iff layer_input[i]<threhshold
        mask_value = K.relu(K.sign(layer_input - self.layer.threshold))
        if self.layer.max_value:
            mask_value *= K.relu(K.sign(self.layer.max_value - layer_input))

        return layer_output*(self.layer.negative_slope*mask_neg_slope + (1-mask_neg_slope)*mask_value)
    


def get_backward_ReLU(layer:ReLU, use_bias=True) -> Layer:
    return BackwardReLU(layer, use_bias)

