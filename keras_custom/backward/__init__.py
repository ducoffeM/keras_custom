from .convolutional import get_backward_Conv2D
from .custom import get_backward_MulConstant
from .normalization import get_backward_BatchNormalization
from .reshaping import get_backward_ZeroPadding2D
from .pooling import get_backward_AveragePooling2D

from .convert import get_backward