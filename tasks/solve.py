from keras.layers import Input
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
        encoder_state = encoder_model.outputs[5]

        n_read_heads = 1
        n_write_heads = 1
        layer = NTMLayer(self.out_dim, self.tm_state_units,
                         n_read_heads, n_write_heads,
                         self.is_cam, self.num_shift,
                         self.N, self.M,
                         return_sequences=self.ret_seq, return_state=False,
                         name='NTM_Layer_Solver')

        tm_aux_seq = Input(shape=(None, self.aux_in_dim))
        input_state_solve = [Input(shape=(s,)) for s in layer.state_size[:-1]]
        init_state = input_state_solve + [encoder_state]
        tm_output_seq = layer(tm_aux_seq, initial_state=init_state)

        ntm_inputs = [tm_input_seq, tm_aux_seq] + input_state_encoder + input_state_solve
        ntm_model = Model(inputs=ntm_inputs, outputs=tm_output_seq)
        return ntm_model

    def encoder_freeze_weights(self, weights):
        self.encoder_layer.set_weights(weights)
        self.encoder_layer.trainable = False

    @property
    def encoder_layer(self):
        return self.encoder_model.get_layer(name='NTM_Layer_Encoder')

    @property
    def solver_layer(self):
        return self.model.get_layer(name='NTM_Layer_Solver')

    def init_state(self, batch_size):
        encoder_init_state = self.encoder_layer.init_state(batch_size)
        solver_init_state = self.solver_layer.init_state(batch_size)
        return encoder_init_state, solver_init_state
