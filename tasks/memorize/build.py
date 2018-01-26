import numpy as np
from sacred import Experiment

from tasks.solve import NTM_Solve

# change below based on task ----
TASK_NAME = 'memorize'
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
    in_dim = element_size
    out_dim = element_size + 1
    aux_in_dim = 1
    ret_seq = True
    ntm = NTM_Solve(in_dim, out_dim, aux_in_dim, tm_state_units,
                    ret_seq, is_cam, num_shift, N, M)
    return ntm


@ex.capture
def build_data_gen(ntm, batch_size, min_len, max_len, _rnd):
    mem_data_gen = ntm.mem_data_simple_gen(batch_size, _rnd)
    mem_init_state = next(mem_data_gen)
    recall_init_state = ntm.solve_layer.init_state(batch_size)
    init_state = mem_init_state + recall_init_state[:-1]
    yield init_state

    aux_in_dim = 1
    while True:
        mem_length = _rnd.randint(low=min_len, high=max_len + 1)
        mem_input = mem_data_gen.send(mem_length)
        mem_input = np.insert(mem_input, 0, 0, axis=1)
        mem_input = np.insert(mem_input, 0, 0, axis=2)
        mem_input[:, 0, 0] = 1
        target = mem_input
        aux_seq = np.ones((batch_size, target.shape[1], aux_in_dim)) * 0.5
        inputs = [mem_input, aux_seq]
        yield inputs, target, mem_length

# end change ---
