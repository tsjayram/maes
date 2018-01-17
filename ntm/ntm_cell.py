from keras.layers import Layer
from keras.layers import Reshape, Flatten

from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead
from ntm.controller import Controller


class NTMCell(Layer):
    def __init__(self, tm_output_units, output_trainable, tm_state_units,
                 n_read_heads, n_write_heads, is_cam,
                 N, M, name='NTMCell'):
        super(NTMCell, self).__init__(name=name)

        self.tm_output_units = tm_output_units
        self.output_trainable = output_trainable
        self.tm_state_units = tm_state_units
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.is_cam = is_cam
        self.N = N
        self.M = M
        self.read_head = None
        self.write_head = None
        self.controller = None
        self._state_size = [('out', self.tm_output_units), ('state', self.tm_state_units),
                            ('read', self.n_read_heads * N), ('write', self.n_write_heads * N),
                            ('memory', N * M)]

    def build(self, input_shape):        # TM input shape
        super(NTMCell, self).build(input_shape)
        # build the read/write heads and controller
        tm_in_dim = input_shape[-1]
        self.read_head = ReadHead(self.n_read_heads, self.tm_state_units,
                                  self.M, self.is_cam)
        self.write_head = WriteHead(self.n_write_heads, tm_in_dim, self.tm_state_units,
                                    self.M, self.is_cam)
        self.controller = Controller(self.tm_output_units, self.output_trainable, tm_in_dim,
                                     self.tm_state_units, self.n_read_heads, self.M)

        models = [self.read_head, self.write_head, self.controller]
        self.trainable_weights = [wt for model in models for wt in model.trainable_weights]

    def call(self, tm_input, states): # states = [tm_output, tm_state, wt_read_flat, wt_write_flat, mem_t_flat]
        _, tm_state, wt_read_flat, wt_write_flat, mem_t_flat = states   # ignore previous TM output
        wt_read = Reshape((self.n_read_heads, self.N))(wt_read_flat)
        wt_write = Reshape((self.n_write_heads, self.N))(wt_write_flat)
        mem_t = Reshape((self.N, self.M))(mem_t_flat)
        memory = Memory(mem_t)

        _, wt_write, memory = self.write_head.call(tm_input, tm_state, [wt_write, memory])
        head_data, wt_read = self.read_head.call(tm_state, [wt_read, memory])
        tm_output, tm_state = self.controller.call(tm_input, tm_state, head_data)

        states_flat = [Flatten()(tensor) for tensor in [wt_read, wt_write, memory.tensor]]
        return tm_output, [tm_output, tm_state] + states_flat

    @property
    def state_size(self):
        return tuple(size for _, size in self._state_size)
