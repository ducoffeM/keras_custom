from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer
from keras.models import Sequential
from keras_custom.backward.utils import compute_output_pad, pooling_layer2D

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
    dico_conv['padding'] = 'valid'
    """
    if layer.padding=='same' and max(layer.strides)>1:
        dico_conv['padding'] = 'valid'
    """

    layer_backward = Conv2DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel
    if use_bias:
        layer_backward.bias = layer.bias
    
    layer_backward.built = True

    input_shape_wo_batch = list(layer.input.shape[1:])
    input_shape_wo_batch_wo_pad = list(layer_backward(layer.output)[0].shape)
    # chevauchement cooperatif

    if layer.data_format=="channels_first":
        w_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1]
        h_pad = input_shape_wo_batch[2] - input_shape_wo_batch_wo_pad[2]
    else:
        w_pad = input_shape_wo_batch[0] - input_shape_wo_batch_wo_pad[0]
        h_pad = input_shape_wo_batch[1] - input_shape_wo_batch_wo_pad[1] 


    pad_layers = pooling_layer2D(w_pad, h_pad, layer.data_format)
    if len(pad_layers):
        layer_backward = Sequential([layer_backward]+pad_layers)
        _= layer_backward(layer.output)
    return layer_backward



