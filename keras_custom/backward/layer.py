# abstract class for BackwardLayer
from keras.layers import Layer, Wrapper


class BackwardLayer(Wrapper):
    """
    A custom Keras wrapper layer that reverses the operations of a given layer.

    `BackwardLayer` is designed to perform the reverse operation of a wrapped Keras
    layer during the backward pass, which can be useful for certain architectures
    requiring layer operations to be undone or applied in reverse.

    Usage:
    ------
    ```python
    from keras.layers import Dense
    backward_layer = BackwardLayer(layer=Dense(32), use_bias=False)
    ```

    This layer can be useful in advanced neural network architectures that
    need custom layer reversals, such as autoencoders, invertible networks,
    or specialized backpropagation mechanisms.

    Notes:
    ------
    - The `BackwardLayer` does not automatically compute a true inverse; it only
      reverses the application of operations as defined by the wrapped layer.
    - It requires the wrapped layer to have compatible reverse operations.
    """

    def __init__(
        self,
        layer: Layer,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.use_bias = use_bias
