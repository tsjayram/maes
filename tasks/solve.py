import numpy as np
from keras.layers import Input
from keras.models import Model

from model.ntm import NTM
from ntm.ntm_layer import NTMLayer


class NTM_Solve(NTM):
    def __init__(self, in_dim, out_dim, aux_in_dim, tm_state_units,
                 ret_seq, is_cam, num_shift,
                 N, M,
                 name='NTM_Solve'):

        super(NTM_Solve, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aux_in_dim = aux_in_dim
        self.tm_state_units = tm_state_units
        self.ret_seq = ret_seq
        self.is_cam = is_cam
        self.num_shift = num_shift
        self.N = N
        self.M = M
        self.name = name
        self._mem_model = None

    def _build_mem_model(self):
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
                         name='NTM_Layer_Memorize')
        tm_input_seq = Input(shape=(None, self.in_dim + 1))
        input_state = [Input(shape=(s,)) for s in layer.state_size]
        ntm_outputs = layer(tm_input_seq, initial_state=input_state)
        ntm_inputs = [tm_input_seq] + input_state
        self._mem_model = Model(inputs=ntm_inputs, outputs=ntm_outputs)
        return self._mem_model

    @property
    def mem_model(self):
        if self._mem_model is None:
            self._mem_model = self._build_mem_model()
        return self._mem_model

    def _build_model(self):
        mem_model = self.mem_model
        tm_input_seq = mem_model.inputs[0]
        input_state_mem = mem_model.inputs[1:]
        mem_state = mem_model.outputs[5]

        n_read_heads = 1
        n_write_heads = 1
        layer = NTMLayer(self.out_dim, self.tm_state_units,
                         n_read_heads, n_write_heads,
                         self.is_cam, self.num_shift,
                         self.N, self.M,
                         return_sequences=self.ret_seq, return_state=False,
                         name='NTM_Layer_Solve')

        tm_aux_seq = Input(shape=(None, self.aux_in_dim))
        input_state_recall = [Input(shape=(s,)) for s in layer.state_size[:-1]]
        init_state = input_state_recall + [mem_state]
        tm_output_seq = layer(tm_aux_seq, initial_state=init_state)

        ntm_inputs = [tm_input_seq, tm_aux_seq] + input_state_mem + input_state_recall
        ntm_model = Model(inputs=ntm_inputs, outputs=tm_output_seq)
        return ntm_model

    @property
    def mem_layer(self):
        return self.mem_model.get_layer(name='NTM_Layer_Memorize')

    @property
    def solve_layer(self):
        return self.model.get_layer(name='NTM_Layer_Solve')

    def mem_freeze_weights(self, weights):
        self.mem_layer.set_weights(weights)
        self.mem_layer.trainable = False

    def mem_data_gen(self, batch_size, min_len, max_len, rnd):
        init_state = self.mem_layer.init_state(batch_size)

        while True:
            length = rnd.randint(low=min_len, high=max_len + 1)
            target = rnd.binomial(1, 0.5, (batch_size, length, self.in_dim))
            inp = np.empty((batch_size, length + 1, self.in_dim + 1))
            inp[:, :length, :self.in_dim] = target
            # markers
            inp[:, :length, self.in_dim] = np.zeros((length,))
            inp[:, length, :self.in_dim] = np.zeros((self.in_dim,))
            inp[:, length, self.in_dim] = 1
            yield inp, init_state, length
