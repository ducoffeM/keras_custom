import os
import numpy as np
from numpy.testing import assert_almost_equal
import keras
from keras.models import Sequential
import torch



def is_invertible(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    batch_size = 10
    input_random = np.reshape(np.random.rand(np.prod(input_shape) * batch_size), [batch_size] + input_shape)
    model = Sequential([layer, backward_layer])
    output_random = model.predict(input_random)
    assert_almost_equal(input_random, output_random)


def linear_mapping(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    output_shape = list(layer.output.shape[1:])
    n_input = np.prod(input_shape)
    n_output = np.prod(output_shape)
    weights_in = np.reshape(np.eye(n_input), [n_input] + input_shape)

    model_layer = Sequential([layer])
    model_backward = Sequential([backward_layer])
    w_f = model_layer.predict(weights_in, verbose=0)
    w_f = np.reshape(w_f, [n_input, n_output])

    weights_out = np.reshape(np.eye(n_output), [n_output] + output_shape)
    w_b = model_backward.predict(weights_out, verbose=0)
    w_b = np.reshape(w_b, [n_output, n_input])
    w_b = w_b.T
    assert_almost_equal(w_f, w_b)


def serialize(layer, backward_layer):

    input_shape = list(layer.input.shape[1:])
    batch_size = 10
    input_random = np.reshape(np.random.rand(np.prod(input_shape) * batch_size), [batch_size] + input_shape)
    toy_model = Sequential([layer, backward_layer])

    filename = "test_serialize_{}_{}.keras".format(layer.__class__.__name__, layer.name)
    # detach toy model to cpu
    # toy_model.to('cpu')
    toy_model.save(filename)  # The file needs to end with the .keras extension
    output_before_export = toy_model.predict(input_random)

    # deserialize
    load_model = keras.models.load_model(filename)

    # compare with the previous output
    output_after_export = load_model.predict(input_random)
    os.remove(filename)
    try:
        np.testing.assert_almost_equal(output_before_export, output_after_export, err_msg="corrupted weights")
    except:
        import pdb; pdb.set_trace()

def compute_backward(input_shape, model, backward_model):
    input_ = torch.ones((1, input_shape[0]), requires_grad=True)
    #input_ = torch.randn(1, input_shape[0], requires_grad=True)
    output = model(input_)
    select_output = output[0,0]
    select_output.backward()
    gradient = input_.grad.cpu().detach().numpy()

    mask_output = torch.Tensor([1]+[0]*31)[None]

    gradient_ = backward_model([mask_output, input_]).cpu().detach().numpy()
    import pdb; pdb.set_trace()



