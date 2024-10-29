import logging
from typing import Any, Optional

from keras_custom.layers import MulConstant
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
    get_backward_ZeroPadding2D,
    get_backward_AveragePooling2D
)

logger = logging.getLogger(__name__)

BACKWARD_PREFIX = "get_backward"

default_mapping_keras2backward_layer: dict[type[Layer], type[callable]]={
    Conv2D: get_backward_Conv2D,
    ZeroPadding2D: get_backward_ZeroPadding2D,
    BatchNormalization: get_backward_BatchNormalization,
    #pooling
    AveragePooling2D:get_backward_AveragePooling2D,
    # custom
    MulConstant: get_backward_MulConstant
}
"""Default mapping between keras layers and get_backward callable"""

def get_backward(layer:Layer, use_bias:bool=True):
    keras_class = type(layer)
    get_backward_layer = default_mapping_keras2backward_layer.get(keras_class)
    return get_backward_layer(layer, use_bias)
