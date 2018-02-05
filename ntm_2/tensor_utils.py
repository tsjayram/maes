import numpy as np
from keras import backend as K

# Note: the shape of "batch" is a common prefix to the shapes of all the arguments


# Batch normalize last dimension with fuzzy factor
def normalize(x):
    return x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())


# Batch similarity
def sim(q, data, normalize=False, adjoint=False):
    # q.shape = batch_shape x h x N, if adjoint is True; q.shape = batch_shape x h x M, if adjoint is False
    # data.shape = batch_shape x N x M. If adjoint is True then we should permute last 2 dimensions
    # out[...,i,j] = sum_k q[...,i,k] * data[...,j,k] for the default options
    assert K.ndim(q) == K.ndim(data)
    q_axis = K.ndim(q) - 1
    data_axis = K.ndim(data) - 2 if adjoint else K.ndim(data) - 1
    if normalize:
        q = K.l2_normalize(q + K.epsilon(), axis=q_axis)
        data = K.l2_normalize(data + K.epsilon(), axis=data_axis)
    return K.batch_dot(q, data, axes=[q_axis, data_axis])


# Batch 1D convolution of unequal length vectors
def circular_conv(x, filter):
    # computes out[...,i] = sum_{j=-ceil(s/2)+1}^{floor(s/2)} x[...,i-j] * filter[...,j]
    s = K.int_shape(filter)[-1]
    N = K.int_shape(x)[-1]
    assert s <= N
    indices = list(range(s//2+1)) + list(range(-s//2+1,0))
    cyclic = np.asfarray([np.roll(np.eye(N), j, axis=0) for j in indices])
    filter_rotate = K.sum(cyclic * filter[...,None,None], axis=-3)
    ans = sim(filter_rotate, x[..., None], adjoint=True)
    return K.squeeze(ans, axis=-1)


# Batch outer product of two vectors
def outer_prod(x, y):
    # K.expand_dims(x, axis=-1) * K.expand_dims(y, axis=-2)
    # sim(x[..., :, None], y[..., :, None])
    return x[..., :, None] * y[..., None, :]
