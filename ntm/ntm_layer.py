import numpy as np
from keras.layers import RNN
from ntm.ntm_cell import NTMCell


class NTMLayer(RNN):
    def __init__(self, tm_output_units, tm_state_units,
                 n_read_heads, n_write_heads, is_cam, N, M,
                 return_sequences=True, return_state=False,
                 stateful=False,
                 name='NTMLayer'):
        if tm_output_units > 0:
            self.tm_output_units = tm_output_units
            self.output_trainable = True
        else:
            self.tm_output_units = 1
            self.output_trainable = False

        self.tm_state_units = tm_state_units
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.is_cam = is_cam
        self.N = N
        self.M = M

        self.ntm_cell = NTMCell(self.tm_output_units, self.output_trainable,
                                self.tm_state_units,
                                self.n_read_heads, self.n_write_heads, self.is_cam,
                                self.N, self.M)
        super(NTMLayer, self).__init__(self.ntm_cell, return_sequences=return_sequences,
                                       return_state=return_state, stateful=stateful,
                                       name=name)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.tm_output_units)
        else:
            output_shape = (input_shape[0], self.tm_output_units)

        if self.return_state:
            return [output_shape] + [(input_shape[0], dim) for dim in self.cell.state_size]
        else:
            return output_shape

    def init_state(self, batch_size):
        tm_output = np.zeros((batch_size, self.tm_output_units))
        tm_state = np.ones((batch_size, self.tm_state_units)) * 0.01

        wt_read = np.zeros((batch_size, self.n_read_heads, self.N))
        wt_read[:, :, 0] = 1.0
        wt_read_flat = np.reshape(wt_read, (batch_size, self.n_read_heads * self.N))

        wt_write = np.zeros((batch_size, self.n_write_heads, self.N))
        wt_write[:, :, 0] = 1.0
        wt_write_flat = np.reshape(wt_write, (batch_size, self.n_write_heads * self.N))

        mem_t_flat = np.ones((batch_size, self.N * self.M)) * 0.01

        return [tm_output, tm_state, wt_read_flat, wt_write_flat, mem_t_flat]

    @property
    def state_size(self):
        return self.ntm_cell.state_size
