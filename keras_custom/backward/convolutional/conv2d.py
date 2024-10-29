from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer

def get_backward_Conv2D(layer: Conv2D, use_bias=True) -> Layer:

    dico_conv = layer.get_config()
    dico_conv.pop("groups")
    input_shape = list(layer.input.shape[1:])
    # update filters to match input, pay attention to data_format
    if layer.data_format == "channels_first":  # better to use enum than raw str
        dico_conv["filters"] = input_shape[0]
    else:
        dico_conv["filters"] = input_shape[-1]

    dico_conv["use_bias"] = use_bias

    layer_backward = Conv2DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel
    if use_bias:
        layer_backward.bias = layer.bias
    
    layer_backward.built = True

    return layer_backward