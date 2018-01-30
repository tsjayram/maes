#!/usr/bin/env python
import os
import arrow

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
matplotlib.rcParams['image.interpolation'] = 'nearest'

import h5py
from sacred import Experiment

from tasks.encode.plot.encoder import NTMEncoder
from tasks.encode.plot.plot_ntm_run import plot_ntm_run

TASK_NAME = 'encode'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../../logs/'

time_str = '2018-01-20__10_46_34_AM'
# time_str = '2018-01-28__12_40_36_AM'

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
    N = 128
    batch_size = 1
    length = 64
    bias = 0.5
    epochs = [13151]
    # epochs = [23440]


@ex.capture
def make_plots_dir(seed, length, N, bias):
    format_str = '/plots_seed={}_L={:04d}_N={:04d}_bias={:04f}'
    plots_dir = PLOTS_ROOT + format_str.format(seed, length, N, bias)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


@ex.capture
def build_ntm(element_size, N, M):
    batch_size = 1
    in_dim = element_size + 1
    ntm = NTMEncoder(batch_size, in_dim, N, M)
    return ntm


@ex.capture
def get_input(length, bias, element_size, _rnd):
    batch_size = 1

    inp = _rnd.binomial(1, bias, (batch_size, length, element_size))

    # marker
    pad = np.zeros((batch_size, 1, element_size))
    inp = np.append(inp, pad, axis=1)
    pad = np.zeros((batch_size, length+1, 1))
    inp = np.append(inp, pad, axis=2)
    inp[: -1, -1] = 1

    # inp = np.empty((batch_size, length + 1, element_size))
    # inp[:, -1, :] = 0
    # inp[:, :, -1] = 0
    # inp[:, -1, -1] = 1
    # inp[:, :-1, :-1] = _rnd.binomial(1, bias, (batch_size, length, element_size-1))

    # control channel
    # inp = np.insert(inp, 0, 0, axis=1)
    # inp = np.insert(inp, 0, 0, axis=2)
    # inp[:, 0, 0] = 1

    return inp


@ex.automain
def run(epochs, seed):
    plots_dir = make_plots_dir(seed)
    ntm = build_ntm()
    # print(ntm.pretty_print_str())
    inp = get_input()
    with h5py.File(MODEL_WTS, 'r') as f:
        epoch_keys = ['epoch_{:05d}'.format(num) for num in epochs]
        time_now = arrow.now().format('YYYY-MM-DD__hh_mm_ss_A')
        for key in (x for x in epoch_keys if x in f):
            print('In ', key)
            grp = f[key]
            weights = [grp[name] for name in grp if 'Encoder' in name]
            ntm.set_weights(weights)
            ntm_run_data = ntm.get_run_data(inp)
            print(ntm_run_data['memory'][0, -1, :, :])
            # print(ntm_run_data['write'][0, :, 0, :])
            # fig = plot_ntm_run(inp, ntm_run_data)
            # filename = '/fig_{}_{}.pdf'.format(key, time_now)
            # fig.savefig(plots_dir + filename, bbox_inches='tight')
            # fig.show()
            # matplotlib.pyplot.pause(1000)
            # matplotlib.pyplot.close(fig)
