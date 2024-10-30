import logging
from typing import Any, Optional

from keras_custom.layers import MulConstant
from keras_custom.backward import BackwardLayer
from keras.layers import (
    Activation,
    Add,
    Average,
    Subtract,
    Dense,
    Layer,
    Conv2D,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Flatten,
    RepeatVector,
    Reshape,
    Permute,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    BatchNormalization,
    GroupNormalization,
    UnitNormalization,
    LayerNormalization,
    SpectralNormalization,
    LeakyReLU,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    MaxPooling2D,
    Dropout,
)

from keras_custom.backward import (
    get_backward_BatchNormalization,
    get_backward_Conv2D,
    get_backward_MulConstant,
    get_backward_ZeroPadding1D,
    get_backward_ZeroPadding2D,
    get_backward_ZeroPadding3D,
    get_backward_Cropping1D,
    get_backward_Cropping2D,
    get_backward_Cropping3D,
    get_backward_Flatten,
    get_backward_Reshape,
    get_backward_Permute,
    get_backward_RepeatVector,
    get_backward_AveragePooling2D,
    get_backward_AveragePooling1D,
    get_backward_AveragePooling3D,
    get_backward_GlobalAveragePooling2D,
    get_backward_GlobalAveragePooling1D,
    get_backward_GlobalAveragePooling3D,
)

logger = logging.getLogger(__name__)

BACKWARD_PREFIX = "get_backward"

default_mapping_keras2backward_layer: dict[type[Layer], type[callable]]={
    # convolution
    Conv2D: get_backward_Conv2D,
    # reshaping
    ZeroPadding1D: get_backward_ZeroPadding1D,
    ZeroPadding2D: get_backward_ZeroPadding2D,
    ZeroPadding3D: get_backward_ZeroPadding3D,
    Cropping1D: get_backward_Cropping1D,
    Cropping2D: get_backward_Cropping2D,
    Cropping3D: get_backward_Cropping3D,
    Flatten: get_backward_Flatten,
    Reshape: get_backward_Reshape,
    Permute: get_backward_Permute,
    RepeatVector: get_backward_RepeatVector,
    # normalization
    BatchNormalization: get_backward_BatchNormalization,
    #pooling
    AveragePooling2D:get_backward_AveragePooling2D,
    AveragePooling1D:get_backward_AveragePooling1D,
    AveragePooling3D:get_backward_AveragePooling3D,
    GlobalAveragePooling2D:get_backward_GlobalAveragePooling2D,
    GlobalAveragePooling1D:get_backward_GlobalAveragePooling1D,
    GlobalAveragePooling3D:get_backward_GlobalAveragePooling3D,
    # custom
    MulConstant: get_backward_MulConstant
}
"""Default mapping between keras layers and get_backward callable"""

def get_backward(layer:Layer, use_bias:bool=True):
    keras_class = type(layer)
    if isinstance(layer, BackwardLayer):
        if use_bias:
            return layer.layer
        elif not hasattr(layer.layer, 'use_bias'):
            return layer.layer
        else:
            # copy layer
            config = layer.layer.get_config()
            config['use_bias']=False
            layer_ = layer.layer.__class__.from_config(config)
            layer_.weights = layer.layer.weights[:1]
            layer_.built=True
            return layer_
            
    get_backward_layer = default_mapping_keras2backward_layer.get(keras_class)
    return get_backward_layer(layer, use_bias)
