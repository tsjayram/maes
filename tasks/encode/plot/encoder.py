import numpy as np
from keras.layers import Input
from keras.models import Model

from model.ntm import NTM
from ntm.ntm_layer import NTMLayer


class NTMEncoder(NTM):
    def __init__(self, batch_size, in_dim, N, M, name='NTM_Encoder'):
        super(NTMEncoder, self).__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.N = N
        self.M = M
        self.name = name

    def _build_model(self):
        tm_state_units = 3
        n_read_heads = 1
        n_write_heads = 1
        is_cam = False
        num_shift = 3
        layer = NTMLayer(0, tm_state_units,
                         n_read_heads, n_write_heads,
                         is_cam, num_shift,
                         self.N, self.M,
                         return_sequences=False, return_state=True,
                         stateful=True,
                         name='NTM_Layer_Encoder')
        tm_input_seq = Input(batch_shape=(self.batch_size, None, self.in_dim))
        ntm_outputs = layer(tm_input_seq)
        model = Model(inputs=tm_input_seq, outputs=ntm_outputs)
        return model

    @property
    def layer(self):
        return self.model.get_layer(name='NTM_Layer_Encoder')

    def set_weights(self, weights):
        self.layer.set_weights(weights)
        self.layer.trainable = False

    def get_run_data(self, inp):
        bs = self.batch_size
        length = inp.shape[1]
        n_read_heads = 1
        n_write_heads = 1
        run_data = {
            'read': np.zeros((bs, length+1, n_read_heads, self.N)),
            'write': np.zeros((bs, length+1, n_write_heads, self.N)),
            'memory': np.zeros((bs, length+1, self.N, self.M)),
        }

        init_state = self.layer.init_state(bs)
        self.layer.reset_states(states=init_state)
        run_data['read'][:, 0, ...] = init_state[2].reshape((bs, n_read_heads, self.N))
        run_data['write'][:, 0, ...] = init_state[3].reshape((bs, n_write_heads, self.N))
        run_data['memory'][:, 0, ...] = init_state[4].reshape((bs, self.N, self.M))

        for j in range(length):
            if j % 20 == 0:
                print('Index=', j)
            all_out = self.model.predict(inp[:, [j], :], batch_size=self.batch_size)
            run_data['read'][:, j+1, ...] = all_out[3].reshape((bs, n_read_heads, self.N))
            run_data['write'][:, j+1, ...] = all_out[4].reshape((bs, n_write_heads, self.N))
            run_data['memory'][:, j+1, ...] = all_out[5].reshape((bs, self.N, self.M))

        return run_data
