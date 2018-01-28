import numpy as np
from sacred import Experiment

from tasks.solver import NTM_Solve

# change below based on task ----
TASK_NAME = 'mem_distract'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../logs/'


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 8
    num_shift = 3
    is_cam = True
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
def build_data_gen(ntm, batch_size, min_num_seq, max_num_seq, min_len, max_len, _rnd):
    mem_data_gen = ntm.mem_data_simple_gen(batch_size, _rnd)
    mem_init_state = next(mem_data_gen)
    recall_init_state = ntm.solve_layer.init_state(batch_size)
    init_state = mem_init_state + recall_init_state[:-1]
    yield init_state
    aux_in_dim = 1
    while True:
        num_seq = _rnd.randint(low=min_num_seq, high=max_num_seq + 1)
        lengths = _rnd.randint(low=min_len, high=max_len + 1, size=(num_seq,))
        lengths = lengths + 3
        mem_length = np.sum(lengths)
        mem_input = mem_data_gen.send(np.sum(lengths))

        sa = np.zeros((mem_input.shape[2],))
        sa[-1] = 1
        so = np.ones_like(sa)

        indices = np.cumsum(lengths)
        indices = np.insert(indices[:-1], 0, [0])

        mem_input[:, indices, :] = sa
        mem_input[:, indices+2, :] = so

        sub_arrays = np.split(mem_input, indices[1:], axis=1)
        target = np.concatenate(sub_arrays[0::2], axis=1)
        target = np.concatenate([target, sub_arrays[-1]], axis=1)

        aux_seq = np.ones((batch_size, target.shape[1], aux_in_dim)) * 0.5
        inputs = [mem_input, aux_seq]
        init_state = mem_init_state + recall_init_state[:-1]
        yield inputs, init_state, target, mem_length

# end change ---
