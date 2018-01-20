"""NTM Read/Write Head."""
from keras.layers import Dense, Input, Reshape, Concatenate
from keras.models import Model
from keras.activations import softplus, sigmoid, softmax, tanh

from ntm.tensor_utils import circular_conv, normalize


class Head:
    def _model2d(self, input_shape, params, name):
        x = Input(input_shape)
        y = [Reshape((self.n_heads, n))
             (Dense(self.n_heads * n, activation=a, name=name + '_' + s)(x))
             for s, (n, a) in params]
        return Model(inputs=x, outputs=y)

    def __init__(self, n_heads, tm_state_units, is_cam, num_shift, M, name='Read'):
        self.n_heads = n_heads
        self.is_cam = is_cam
        # build a controller for all the heads given size and activation for each parameter
        params = [('s', (num_shift, lambda z: softmax(softplus(z)))),
                  ('gamma', (1, lambda z: 1 + softplus(z)))]
        if self.is_cam:
            params = [('k', (M, tanh)), ('beta', (1, softplus)), ('g', (1, sigmoid))] + params

        self.head_ctrl = self._model2d((tm_state_units,), params, name)
        self.trainable_weights = self.head_ctrl.trainable_weights

    def call(self, tm_state, states):                  # states = [weights, memory]
        wt, memory = states

        if self.is_cam:
            k, β, g, s, γ = self.head_ctrl(tm_state)       # extract parameters from controller
            wt_k = memory.content_similarity(k)            # content addressing ...
            wt_β = softmax(β * wt_k)                       # ... modulated by β
            wt = g * wt_β + (1 - g) * wt                   # scalar interpolation
        else:
            s, γ = self.head_ctrl(tm_state)                # extract parameters from controller

        wt_s = circular_conv(wt, s)                      # convolution with shift
        wt = normalize(wt_s**γ)                          # sharpening with normalization

        head_data = memory.address_read(wt)
        return [head_data, wt]


ReadHead = Head


class WriteHead(Head):
    def __init__(self, n_heads, tm_in_dim, tm_state_units, is_cam, num_shift, M, name='Write'):
        super(WriteHead, self).__init__(n_heads, tm_state_units, is_cam, num_shift, M, name)
        params = [('e', (M, sigmoid)), ('a', (M, None))]
        self.head_ctrl_write = self._model2d((tm_in_dim + tm_state_units,), params, name)
        self.trainable_weights = self.trainable_weights + self.head_ctrl_write.trainable_weights

    def call(self, tm_input, tm_state, states):                     # states = [weights, memory]
        wt, memory = states
        erase, add = self.head_ctrl_write(Concatenate()([tm_input, tm_state]))
        memory.address_erase(wt, erase)
        memory.address_add(wt, add)
        return super(WriteHead, self).call(tm_state, [wt, memory]) + [memory]
