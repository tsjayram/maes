from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model

from model.ntm import NTM
from ntm.ntm_layer import NTMLayer


class NTMSolver(NTM):
    def __init__(self, in_dim, out_dim, aux_in_dim, tm_state_units,
                 ret_seq, is_cam, num_shift,
                 N, M,
                 name='NTM_Solver'):

        super(NTMSolver, self).__init__()

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
        self._encoder_model = None
        self.solver_state_indices_fwd = [0, 1, 2, 3]  # everything except mem
        self.solver_state_indices_rev = [0, 1, 3]  # everything except read and mem

    def _build_encoder_model(self):
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
                         name='NTM_Layer_Encoder')
        tm_input_seq = Input(shape=(None, self.in_dim))
        input_state = [Input(shape=(s,)) for s in layer.state_size]
        ntm_outputs = layer(tm_input_seq, initial_state=input_state)
        ntm_inputs = [tm_input_seq] + input_state
        self._encoder_model = Model(inputs=ntm_inputs, outputs=ntm_outputs)
        return self._encoder_model

    @property
    def encoder_model(self):
        if self._encoder_model is None:
            self._encoder_model = self._build_encoder_model()
        return self._encoder_model

    def _build_model(self):
        encoder_model = self.encoder_model
        tm_input_seq = encoder_model.inputs[0]
        input_state_encoder = encoder_model.inputs[1:]
        encoder_state_write = encoder_model.outputs[-2]  # write attention ...
        encoder_state_mem = encoder_model.outputs[-1]    # ... and memory

        tm_aux_seq = Input(shape=(None, self.aux_in_dim))

        n_read_heads = 1
        n_write_heads = 1
        layer_fwd = NTMLayer(self.out_dim, self.tm_state_units,
                             n_read_heads, n_write_heads,
                             self.is_cam, self.num_shift,
                             self.N, self.M,
                             return_sequences=self.ret_seq, return_state=False,
                             name='NTM_Layer_Solver_Fwd')

        sizes = [layer_fwd.state_size[j] for j in self.solver_state_indices_fwd]
        input_state_solve_fwd = [Input(shape=(s,)) for s in sizes]
        init_state_fwd = list(input_state_solve_fwd)  # make a shallow copy
        init_state_fwd.append(encoder_state_mem)  # mem state of encoder => mem state of fwd solver

        tm_output_seq_fwd = layer_fwd(tm_aux_seq, initial_state=init_state_fwd)

        layer_rev = NTMLayer(self.out_dim, self.tm_state_units,
                             n_read_heads, n_write_heads,
                             self.is_cam, self.num_shift,
                             self.N, self.M,
                             return_sequences=self.ret_seq, return_state=False,
                             name='NTM_Layer_Solver_Rev')

        sizes = [layer_rev.state_size[j] for j in self.solver_state_indices_rev]
        input_state_solve_rev = [Input(shape=(s,)) for s in sizes]
        init_state_rev = list(input_state_solve_rev)  # make a shallow copy
        init_state_rev.insert(2, encoder_state_write)  # write of encoder => read attn of solver
        init_state_rev.append(encoder_state_mem)  # mem state of encoder => mem state of solver

        tm_output_seq_rev = layer_rev(tm_aux_seq, initial_state=init_state_rev)

        ntm_state_inputs = input_state_encoder + input_state_solve_fwd + input_state_solve_rev
        ntm_inputs = [tm_input_seq, tm_aux_seq] + ntm_state_inputs
        ntm_outputs = concatenate([tm_output_seq_fwd, tm_output_seq_rev], axis=1)
        ntm_model = Model(inputs=ntm_inputs, outputs=ntm_outputs)
        return ntm_model

    def encoder_freeze_weights(self, weights):
        self.encoder_layer.set_weights(weights)
        self.encoder_layer.trainable = False

    @property
    def encoder_layer(self):
        return self.encoder_model.get_layer(name='NTM_Layer_Encoder')

    @property
    def solver_layer_fwd(self):
        return self.model.get_layer(name='NTM_Layer_Solver_Fwd')

    @property
    def solver_layer_rev(self):
        return self.model.get_layer(name='NTM_Layer_Solver_Rev')

    def init_state(self, batch_size):
        encoder_init_state = self.encoder_layer.init_state(batch_size)

        solver_init_state_fwd = self.solver_layer_fwd.init_state(batch_size)
        solver_init_state_fwd = [solver_init_state_fwd[j] for j in self.solver_state_indices_fwd]

        solver_init_state_rev = self.solver_layer_rev.init_state(batch_size)
        solver_init_state_rev = [solver_init_state_rev[j] for j in self.solver_state_indices_rev]

        for elem in [encoder_init_state, solver_init_state_fwd, solver_init_state_rev]:
            print('Length=', len(elem))
            for array in elem:
                print(array.shape)
        input('pause')
        return encoder_init_state + solver_init_state_fwd + solver_init_state_rev
