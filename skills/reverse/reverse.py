import numpy as np
from keras.layers import Input
from keras.models import Model

from ntm.ntm_layer import NTMLayer
from skills.solve import NTM_Solve


class NTM_Reverse(NTM_Solve):
    def __init__(self, element_size, tm_state_units,
                 is_solve_cam, num_solve_shift,
                 N, M):

        super(NTM_Reverse, self).__init__(element_size, tm_state_units,
                                          is_solve_cam, num_solve_shift,
                                          N, M)

    def _build_model(self):
        mem_model = self.mem_model

        tm_input_seq = mem_model.inputs[0]
        input_state_mem = mem_model.inputs[1:]
        mem_state = mem_model.outputs[5]

        layer = NTMLayer(self.element_size, self.tm_state_units,
                         self.n_read_heads, self.n_write_heads,
                         self.is_solve_cam, self.num_solve_shift,
                         self.N, self.M,
                         return_sequences=True, return_state=False,
                         name='NTM_Layer_Reverse')

        tm_dummy_seq = Input(shape=(None, 1))
        input_state_recall = [Input(shape=(s,)) for s in layer.state_size[:-1]]
        init_state = input_state_recall + [mem_state]
        tm_output_seq = layer(tm_dummy_seq, initial_state=init_state)

        ntm_inputs = [tm_input_seq, tm_dummy_seq] + input_state_mem + input_state_recall
        ntm_model = Model(inputs=ntm_inputs, outputs=tm_output_seq)
        return ntm_model

    @property
    def ntm_memorize(self):
        if self._ntm_mem is None:
            self._ntm_mem = NTM_Memorize(self.element_size, self.tm_state_units,
                                         self.N, self.M)
        return self._ntm_mem

    @property
    def recall_layer(self):
        return self.model.get_layer(name='NTM_Layer_Recall')

    def mem_freeze_weights(self, weights):
        self.ntm_memorize.freeze_weights(weights)

    def data_gen(self, batch_size, min_len, max_len, rnd):
        mem_data_gen = self.ntm_memorize.data_gen(batch_size, min_len, max_len, rnd)
        recall_init_state = self.recall_layer.init_state(batch_size)

        while True:
            mem_inputs, mem_init_state, mem_target, mem_length = next(mem_data_gen)

            mem_target = np.flip(mem_target, axis=1)

            dummy_seq = np.ones((batch_size, mem_target.shape[1], 1)) * 0.5
            inputs = mem_inputs + [dummy_seq]
            init_state = mem_init_state + recall_init_state[:-1]
            yield inputs, init_state, mem_target, mem_length

