from keras.layers import ZeroPadding3D, Cropping3D
from keras.layers import Layer

def get_backward_Cropping3D(layer: Cropping3D, use_bias=True) -> Layer:


    dico_cropping = layer.get_config()
    cropping = dico_cropping['cropping']
    data_format = dico_cropping['data_format']

    layer_backward = ZeroPadding3D(padding=cropping, data_format=data_format)
    layer_backward.built = True

    return layer_backward