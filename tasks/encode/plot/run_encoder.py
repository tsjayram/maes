#!/usr/bin/env python
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams['image.interpolation'] = 'nearest'

import h5py
from sacred import Experiment

from tasks.encode.plot.encoder import NTMEncoder
from tasks.encode.plot.plot_ntm_run import plot_ntm_run

TASK_NAME = 'encode'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../../logs/'

time_str = '2018-01-26__11_56_39_PM'

LOG_DIR = LOG_ROOT + TASK_NAME + '/' + time_str + '/'
MODEL_WTS = LOG_DIR + 'model_weights.hdf5'
PLOTS_ROOT = LOG_DIR + 'plots/'
os.makedirs(PLOTS_ROOT, exist_ok=True)

RANDOM_SEED = 12345


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    num_shift = 3
    is_cam = False
    M = 10


@ex.config
def run_config():
    seed = RANDOM_SEED
    N = 68
    batch_size = 1
    length = 64
    bias = 0.5
    epochs = [7461]


@ex.capture
def make_plots_dir(seed, length, N):
    plots_dir = PLOTS_ROOT + '/plots_seed={}_L={:04d}_N={:04d}'.format(seed, length, N)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


@ex.capture
def build_ntm(element_size, N, M):
    batch_size = 1
    in_dim = element_size + 1
    ntm = NTMEncoder(batch_size, in_dim, N, M)
    return ntm


@ex.capture
def get_seq(length, bias, element_size, _rnd):
    batch_size = 1
    seq = _rnd.binomial(1, bias, (batch_size, length, element_size))
    seq = np.insert(seq, 0, 0, axis=1)
    seq = np.insert(seq, 0, 0, axis=2)

    return seq


@ex.capture
def run_ntm_with_state(ntm, seq, length, N, M):
    n_read_heads = 1
    n_write_heads = 1
    ntm_run_data = {
        'read': np.zeros((length, n_read_heads, N)),
        'write': np.zeros((length, n_write_heads, N)),
        'memory': np.zeros((length, N, M)),
    }

    ntm.reset_states(batch_size=1)
    for j in range(length):
        if j % 20 == 0:
            print('Index=', j)
        ntm_out_with_state = ntm.model.predict(seq[np.newaxis, [j], :], batch_size=1)
        ntm_run_data['read'][j, ...] = ntm_out_with_state[3].reshape((n_read_heads, N))
        ntm_run_data['write'][j, ...] = ntm_out_with_state[4].reshape((n_write_heads, N))
        ntm_run_data['memory'][j, ...] = ntm_out_with_state[5].reshape((N, M))

    return ntm_run_data


@ex.automain
def run(epochs, seed):
    plots_dir = make_plots_dir(seed)

    ntm = build_ntm()
    print(ntm.pretty_print_str())
    seq = get_seq()
    with h5py.File(MODEL_WTS, 'r') as f:
        epoch_keys = ['epoch_{:05d}'.format(num) for num in epochs]
        for key in (x for x in epoch_keys if x in f):
            print('In ', key)
            grp = f[key]
            weights = [grp[name] for name in grp if 'Encoder' in name]
            ntm.set_weights(weights)
            ntm_run_data = run_ntm_with_state(ntm, seq)
            fig = plot_ntm_run(seq, ntm_run_data)
            fig.savefig(plots_dir + '/fig_{}.pdf'.format(key), bbox_inches='tight')
            matplotlib.pyplot.close(fig)
