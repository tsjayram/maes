import numpy as np
from sacred import Experiment

from tasks.n_back.solver import NTMSolver

# change below based on task ----
TASK_NAME = 'n_back'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../logs/'


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    tm_solver_state_units = 3
    num_shift = 3
    is_cam = False
    M = 10


@ex.capture
def build_ntm(element_size, tm_state_units, tm_solver_state_units, is_cam, num_shift, N, M):
    in_dim = element_size + 1
    out_dim = 1
    aux_in_dim = element_size
    ret_seq = True
    ntm = NTMSolver(in_dim, out_dim, aux_in_dim, tm_state_units, tm_solver_state_units,
                    ret_seq, is_cam, num_shift, N, M)
    return ntm


@ex.capture
def build_data_gen(ntm, batch_size, min_len, max_len, bias, element_size, _rnd):
    yield ntm.init_state(batch_size)

    while True:
        seq_length = _rnd.randint(low=min_len, high=max_len + 1)
        seq = _rnd.binomial(1, bias, (batch_size, seq_length, element_size))
        encoder_input = np.insert(seq, 0, 0, axis=1)
        encoder_input = np.insert(encoder_input, 0, 0, axis=2)
        encoder_input[:, 0, 0] = 1

        mask = _rnd.binomial(1, 0.5, (batch_size, seq_length))
        input_xor = _rnd.binomial(1, 0.5, seq.shape)
        input_xor = input_xor * mask[:, :, np.newaxis]
        aux_seq = np.logical_xor(seq, input_xor).astype(float)

        target = np.any(input_xor, axis=2, keepdims=True)

        inputs = [encoder_input, aux_seq]
        yield inputs, target, seq_length

# end change ---
