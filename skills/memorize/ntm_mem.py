import numpy as np
from keras.layers import Input
from keras.models import Model

from ntm.ntm import NTM
from ntm.ntm_layer import NTMLayer


class NTM_Memorize(NTM):
    def __init__(self, element_size, tm_state_units, N, M,
                 name='NTM_Memorize'):

        super(NTM_Memorize, self).__init__()

        self.element_size = element_size
        self.tm_state_units = tm_state_units
        self.N = N
        self.M = M
        self.name = name
        self.n_read_heads = 1
        self.n_write_heads = 1
        self.is_cam = False

    def _build_model(self):
        layer = NTMLayer(0, self.tm_state_units,
                         self.n_read_heads, self.n_write_heads, self.is_cam,
                         self.N, self.M,
                         return_sequences=False, return_state=True,
                         name='NTM_Layer_Memorize')

        tm_input_seq = Input(shape=(None, self.element_size + 1))
        input_state = [Input(shape=(s,)) for s in layer.state_size]
        ntm_outputs = layer(tm_input_seq, initial_state=input_state)
        ntm_inputs = [tm_input_seq] + input_state
        ntm_model = Model(inputs=ntm_inputs, outputs=ntm_outputs)
        return ntm_model

    @property
    def mem_layer(self):
        return self.model.get_layer(name='NTM_Layer_Memorize')

    def data_gen(self, batch_size, min_len, max_len, rnd):
        init_state = self.mem_layer.init_state(batch_size)

        while True:
            length = rnd.randint(low=min_len, high=max_len + 1)
            target = rnd.binomial(1, 0.5, (batch_size, length, self.element_size))
            inp = np.empty((batch_size, length + 1, self.element_size + 1))
            inp[:, :length, :self.element_size] = target
            # markers
            inp[:, :length, self.element_size] = np.zeros((length,))
            inp[:, length, :self.element_size] = np.zeros((self.element_size,))
            inp[:, length, self.element_size] = 1
            yield [inp], init_state, target, length
