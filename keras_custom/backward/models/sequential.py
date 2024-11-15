from keras.layers import Input
from keras.models import Sequential, Model

from keras_custom.backward import get_backward
from keras_custom.backward.layers.layer import BackwardLinearLayer, BackwardNonLinearLayer


def get_backward_sequential(model):

    # convert every layers
    layers_backward = [get_backward(layer, use_bias=False) for layer in model.layers]
    # check if the layers are all linear
    is_linear = min([isinstance(layer_backward, BackwardLinearLayer) for layer_backward in layers_backward])
    if is_linear:
        backward_model = Sequential(layers=layers_backward[::-1])
        # init shape
        backward_model(model.outputs)
        return backward_model
    else:
        # get input_dim without batch
        input_dim_wo_batch = list(model.inputs[0].shape[1:])
        # for output_tensor in model.outputs:
        output_dim_wo_batch = list(model.outputs[0].shape[1:])

        backward_input_tensor = Input(output_dim_wo_batch)
        input_tensor = Input(input_dim_wo_batch)
        # forward propagation
        dico_input_layer = dict()
        output = None
        for layer, backward_layer in zip(model.layers, layers_backward):
            if output is None:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = input_tensor
                output = layer(input_tensor)
            else:
                if isinstance(backward_layer, BackwardNonLinearLayer):
                    dico_input_layer[id(backward_layer)] = output
                output = layer(output)

        gradient = None
        for backward_layer in layers_backward[::-1]:

            if isinstance(backward_layer, BackwardLinearLayer):
                # no need for forward input
                if gradient is None:
                    gradient = backward_layer(backward_input_tensor)
                else:
                    gradient = backward_layer(gradient)
            else:
                input_forward = dico_input_layer[id(backward_layer)]
                if gradient is None:
                    gradient = backward_layer([backward_input_tensor, input_forward])
                else:
                    gradient = backward_layer([gradient, input_forward])


        return Model([input_tensor, backward_input_tensor], gradient)
