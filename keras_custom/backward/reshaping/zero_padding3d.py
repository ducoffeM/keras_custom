from keras.layers import ZeroPadding3D, Cropping3D
from keras.layers import Layer

def get_backward_ZeroPadding3D(layer: ZeroPadding3D, use_bias=True) -> Layer:


    dico_padding = layer.get_config()
    padding = dico_padding['padding']
    data_format = dico_padding['data_format']

    layer_backward = Cropping3D(cropping=padding, data_format=data_format)
    layer_backward.built = True

    return layer_backward