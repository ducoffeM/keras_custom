from keras.layers import DepthwiseConv2D, Conv2DTranspose
from keras.layers import Layer

class BackwardDepthwiseConv2D(Layer):
    """FILL

 
    """

    def __init__(
        self,
        layer:DepthwiseConv2D,
        use_bias:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer = layer
        self.use_bias = use_bias

    def compute_output_shape(self, input_shape):
        return NotImplementedError()
        return self.layer.compute_output_shape(input_shape)

    # serialize ...

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)

        moving_mean = ops.cast(self.layer.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.layer.moving_variance, inputs.dtype)

        if training and self.layer.trainable:
            mean, variance = self.layer._moments(inputs, mask)
        else:
            mean = moving_mean
            variance = moving_variance

        if self.layer.scale:
            gamma = ops.cast(self.layer.gamma, inputs.dtype)
        else:
            gamma = None

        if self.layer.center:
            beta = ops.cast(self.layer.beta, inputs.dtype)
        else:
            beta = None

        # reshape mean, variance, beta, gamme to the right shape
        input_dim_batch = [-1]+[1]*(len(inputs.shape)-1)
        input_dim_batch[self.layer.axis]= inputs.shape[self.layer.axis]

        mean_ = ops.reshape(mean, input_dim_batch)
        variance_ = ops.reshape(variance, input_dim_batch)
        gamma_ = ops.reshape(gamma, input_dim_batch)
        beta_ = ops.reshape(beta, input_dim_batch)

        # z = (x-mean_)/ops.sqrt(variance_+epsilon)
        # inputs = gamma_*z + beta_
        # thus z = (inputs-beta_)/gamma_
        # thus x = z*ops.sqrt(variance_+epsilon) + mean_
        #z = (inputs -beta_)/gamma_
        #x = z*ops.sqrt(variance_+self.layer.epsilon) + mean_
        w = ops.sqrt(variance_+self.layer.epsilon)/gamma_
        outputs = w*inputs

        if self.use_bias:
            b = -beta_*ops.sqrt(variance_+self.layer.epsilon)/gamma_ + mean_
            return outputs+b
        
        return outputs


def depthwise_conv(
        inputs, kernel, strides, padding, data_format, dilation_rate
):
    
    kernel = convert_to_tensor(kernel)
    kernel = torch.reshape(kernel, kernel.shape[:2]+(1, kernel.shape[-2]*kernel.shape[-1]))

    return conv(inputs, kernel, strides, padding, data_format, dilation_rate)

def BackwardDepthwiseConv2D(Layer):



def get_backward_Conv2D(layer: Conv2D, use_bias=True) -> Layer:

    dico_conv = layer.get_config()
    dico_conv.pop("groups")
    input_shape = list(layer.input.shape[1:])
    # update filters to match input, pay attention to data_format
    if layer.data_format == "channels_first":  # better to use enum than raw str
        dico_conv["filters"] = input_shape[0]
    else:
        dico_conv["filters"] = input_shape[-1]

    dico_conv["use_bias"] = use_bias

    layer_backward = Conv2DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel
    if use_bias:
        layer_backward.bias = layer.bias
    
    layer_backward.built = True

    return layer_backward