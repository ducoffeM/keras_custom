from keras.layers import ZeroPadding2D, Cropping2D
from keras.layers import Layer

def get_backward_ZeroPadding2D(layer: ZeroPadding2D, use_bias=True) -> Layer:


    dico_padding = layer.get_config()
    padding = dico_padding['padding']
    data_format = dico_padding['data_format']

    layer_backward = Cropping2D(cropping=padding, data_format=data_format)
    layer_backward.built = True

    return layer_backward