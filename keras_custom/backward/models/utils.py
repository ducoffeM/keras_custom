import keras
from keras_custom.backward.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer


def is_linear(model_backward: keras.models.Model) -> bool:
    return min([isinstance(layer, BackwardLinearLayer) for layer in model_backward.layers])
