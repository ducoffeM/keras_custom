from keras.layers import ZeroPadding1D, Cropping1D
from keras.layers import Layer

def get_backward_ZeroPadding1D(layer: ZeroPadding1D, use_bias=True) -> Layer:


    dico_padding = layer.get_config()
    padding = dico_padding['padding']
    data_format = dico_padding['data_format']

    layer_backward = Cropping1D(cropping=padding, data_format=data_format)
    layer_backward.built = True

    return layer_backward