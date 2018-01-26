import numpy as np
from sacred import Experiment

from tasks.solve import NTMSolve

# change below based on task ----
TASK_NAME = 'encoder'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../logs/'


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    num_shift = 3
    is_cam = False
    M = 10


@ex.capture
def build_ntm(element_size, tm_state_units, is_cam, num_shift, N, M):
    in_dim = element_size + 1
    out_dim = element_size + 1
    aux_in_dim = 1
    ret_seq = True
    ntm = NTMSolve(in_dim, out_dim, aux_in_dim, tm_state_units,
                   ret_seq, is_cam, num_shift, N, M)
    return ntm


@ex.capture
def build_data_gen(ntm, batch_size, min_len, max_len, _rnd):
    encoder_data_gen = ntm.encoder_data_gen(batch_size, _rnd)
    encoder_init_state = next(encoder_data_gen)
    decoder_init_state = ntm.solve_layer.init_state(batch_size)
    init_state = encoder_init_state + decoder_init_state[:-1]
    yield init_state

    aux_in_dim = 1
    while True:
        encoder_length = _rnd.randint(low=min_len, high=max_len + 1)
        encoder_input = encoder_data_gen.send(encoder_length)
        encoder_input[:, :, 0] = 0
        encoder_input = np.insert(encoder_input, 0, 0, axis=1)
        encoder_input[:, 0, 0] = 1
        target = encoder_input
        aux_seq = np.ones((batch_size, target.shape[1], aux_in_dim)) * 0.5
        inputs = [encoder_input, aux_seq]
        yield inputs, target, encoder_length

# end change ---
