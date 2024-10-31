import keras
import keras.ops as K


class Cast(keras.layers.Layer):
    """
    Custom Keras Layer that casts the input tensor to a specified data type.

    This layer allows the conversion of the tensor's data type to various types such as float32, int32, etc.
    The desired data type is specified upon initialization.
    """

    def __init__(self, dtype: int, **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.dtype: int = dtype
        self.cast_map = {
            1: K.float32,
            2: K.uint8,
            3: K.int8,
            5: K.int16,
            6: K.int32,
            7: K.int64,
            9: K.bool,
            10: K.float16,
            11: K.double,
        }

    def call(self, inputs_):
        return keras.ops.cast(inputs_, self.cast_map[self.dtype])

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self.dtype, "cast_map": self.cast_map})
        return config

    def compute_output_shape(self, input_shape):

        return input_shape

    @classmethod
    def from_config(cls, config):
        dtype_config = config.pop("dtype")
        cast_map_config = config.pop("cast_map")
        dtype = keras.saving.deserialize_keras_object(dtype_config)
        cast_map = keras.saving.deserialize_keras_object(cast_map_config)
        return cls(dtype=dtype, cast_map=cast_map, **config)
