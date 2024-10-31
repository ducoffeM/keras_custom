from .convolutional import (
    get_backward_Conv2D,
    get_backward_DepthwiseConv2D,
    get_backward_DepthwiseConv1D,
)
from .custom import get_backward_MulConstant, get_backward_PlusConstant
from .normalization import get_backward_BatchNormalization
from .reshaping import (
    get_backward_ZeroPadding2D,
    get_backward_ZeroPadding3D,
    get_backward_ZeroPadding1D,
    get_backward_Cropping3D,
    get_backward_Cropping1D,
    get_backward_Cropping2D,
    get_backward_Reshape,
    get_backward_Flatten,
    get_backward_Permute,
    get_backward_RepeatVector,
)
from .pooling import (
    get_backward_AveragePooling2D,
    get_backward_AveragePooling3D,
    get_backward_AveragePooling1D,
    get_backward_GlobalAveragePooling2D,
    get_backward_GlobalAveragePooling1D,
    get_backward_GlobalAveragePooling3D,
)

from .convert import get_backward
