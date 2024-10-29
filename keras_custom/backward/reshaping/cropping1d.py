from keras.layers import ZeroPadding1D, Cropping1D
from keras.layers import Layer

def get_backward_Cropping1D(layer: Cropping1D, use_bias=True) -> Layer:


    dico_cropping = layer.get_config()
    cropping = dico_cropping['cropping']
    data_format = dico_cropping['data_format']

    layer_backward = ZeroPadding1D(padding=cropping, data_format=data_format)
    layer_backward.built = True

    return layer_backward