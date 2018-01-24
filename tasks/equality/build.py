import numpy as np
from sacred import Experiment

from tasks.solve import NTM_Solve

# change below based on task ----
TASK_NAME = 'equality'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../permanent_logs/'


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    num_shift = 3
    is_cam = False
    M = 10


@ex.capture
def build_ntm(element_size, tm_state_units, is_cam, num_shift, N, M):
    in_dim = element_size
    out_dim = 1
    aux_in_dim = element_size
    ret_seq = False
    ntm = NTM_Solve(in_dim, out_dim, aux_in_dim, tm_state_units,
                    ret_seq, is_cam, num_shift, N, M)
    return ntm


@ex.capture
def build_data_gen(ntm, batch_size, min_len, max_len, _rnd):
    mem_data_gen = ntm.mem_data_gen(batch_size, min_len, max_len, _rnd)
    recall_init_state = ntm.solve_layer.init_state(batch_size)
    while True:
        mem_input, mem_init_state, mem_length = next(mem_data_gen)
        first_input = mem_input[:, 1:, :-1]
        target_mask = _rnd.binomial(1, 0.5, (batch_size,))
        input_xor = _rnd.binomial(1, 0.5, first_input.shape)
        input_xor = input_xor * target_mask[:, np.newaxis, np.newaxis]
        second_input = np.logical_xor(first_input, input_xor).astype(float)
        target = np.any(input_xor, axis=(1, 2))
        target = target[:, np.newaxis]
        inputs = [mem_input, second_input]
        init_state = mem_init_state + recall_init_state[:-1]
        yield inputs, init_state, target, mem_length

# end change ---
