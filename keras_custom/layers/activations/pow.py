import keras #type:ignore

@keras.saving.register_keras_serializable()
class Pow(keras.layers.Layer):
    """
    Custom Keras Layer that raises each element of the input tensor to a specified power.
    """

    def __init__(self, power: int, **kwargs):
        """
        Initializes the Pow layer with a specified exponent.

        Args:
            power: The exponent to which each element in the input tensor will be raised.
        """
        super(Pow, self).__init__(**kwargs)
        self.power: int = power

    def call(self, inputs_):
        return keras.ops.power(inputs_, self.power)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"power": self.power})
        return config

    @classmethod
    def from_config(cls, config):
        power_config = config.pop("power")
        power = keras.saving.deserialize_keras_object(power_config)
        return cls(power=power, **config)
