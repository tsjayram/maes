import numpy as np
import h5py
import inspect

from sacred import Experiment

from skills.solve import NTM_Solve

LOG_ROOT = '../../logs/'
MEM_FREEZE_WTS = LOG_ROOT + 'memorize/2018-01-16__12_04_47_PM/model_weights.hdf5'

ex = Experiment('Recall')


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    num_shift = 3
    is_cam = False
    M = 10


@ex.config
def mem_weights():
    use_frozen_wts = False
    mem_freeze_wts_file = MEM_FREEZE_WTS
    mem_epoch = 12500


@ex.capture
def build_ntm(element_size, tm_state_units, is_cam, num_shift, N, M):
    in_dim = element_size
    out_dim = element_size
    aux_in_dim = 1
    ret_seq = True
    ntm = NTM_Solve(in_dim, out_dim, aux_in_dim, tm_state_units,
                    ret_seq, is_cam, num_shift, N, M)
    return ntm


@ex.capture
def mem_freeze(ntm, use_frozen_wts, mem_freeze_wts_file, mem_epoch):
    if use_frozen_wts:
        epoch_key = 'epoch_{:05d}'.format(mem_epoch)
        with h5py.File(mem_freeze_wts_file, 'r') as f:
            print('In ', epoch_key)
            grp = f[epoch_key]
            weights = [np.array(grp[name]) for name in grp if 'Memorize' in name]
            ntm.mem_freeze_weights(weights)


@ex.capture
def build_data_gen(ntm, batch_size, min_len, max_len, _rnd):
    mem_data_gen = ntm.mem_data_gen(batch_size, min_len, max_len, _rnd)
    recall_init_state = ntm.solve_layer.init_state(batch_size)
    aux_in_dim = 1
    while True:
        mem_input, mem_init_state, mem_length = next(mem_data_gen)
        target = mem_input[:, :-1, :-1]
        aux_seq = np.ones((batch_size, target.shape[1], aux_in_dim)) * 0.5
        inputs = [mem_input, aux_seq]
        init_state = mem_init_state + recall_init_state[:-1]
        yield inputs, init_state, target, mem_length


def pause():
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    format_str = 'Pause at line {} in function {} of file {}'
    input(format_str.format(info.lineno, info.function, info.filename))
