#!/usr/bin/env python
import os
import numpy as np
import h5py
from sacred import Experiment

from tasks.encode.plot.encoder import NTMEncoder

np.set_printoptions(precision=1, linewidth=250, suppress=True, floatmode='fixed')

TASK_NAME = 'encode'
ex = Experiment(TASK_NAME)
LOG_ROOT = '../../../logs/'

# time_str = '2018-01-20__10_46_34_AM'
# time_str = '2018-01-28__12_40_36_AM'
# time_str = '2018-01-20__06_36_48_PM'
# time_str = '2018-02-01__05_51_53_PM'
time_str = '2018-02-01__06_25_28_PM'

LOG_DIR = LOG_ROOT + TASK_NAME + '/' + time_str + '/'
MODEL_WTS = LOG_DIR + 'model_weights.hdf5'

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
    # epochs = [13151]
    # epochs = [23440]
    # epochs = [18991]
    # epochs = [15524]
    epochs = [16032]


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

    # control channel
    inp = np.insert(inp, 0, 0, axis=1)
    inp = np.insert(inp, 0, 0, axis=2)
    inp[:, 0, 0] = 1
    return inp


@ex.capture
def get_inputs(length, bias, element_size, _rnd):
    batch_size = 1
    seq = _rnd.binomial(1, bias, (batch_size, length, element_size))
    # seq = _rnd.binomial(1, bias, (batch_size, 1, element_size))

    inp = np.empty((batch_size, length, element_size))
    inp[:, :, :] = seq

    inp2 = np.flip(seq, axis=1)

    # inp2 = np.empty((batch_size, length + 1, element_size))
    # inp2[:, :length, :] = seq
    # inp2[:, length, :] = inp[:, 0, :]

    # control channel
    inp = np.insert(inp, 0, 0, axis=1)
    inp = np.insert(inp, 0, 0, axis=2)
    inp[:, 0, 0] = 1

    inp2 = np.insert(inp2, 0, 0, axis=1)
    inp2 = np.insert(inp2, 0, 0, axis=2)
    inp2[:, 0, 0] = 1

    # control channel
    # inp = np.append(inp, np.zeros((batch_size, 1, element_size)), axis=1)
    # inp = np.append(inp, np.zeros((batch_size, length+1, 1)), axis=2)
    # inp[:, -1, -1] = 1
    #
    # inp2 = np.append(inp2, np.zeros((batch_size, 1, element_size)), axis=1)
    # inp2 = np.append(inp2, np.zeros((batch_size, length+2, 1)), axis=2)
    # inp2[:, -1, -1] = 1

    # inp2 = np.copy(inp)
    # inp2[:, -1, :] = _rnd.binomial(1, bias, (batch_size, element_size + 1))
    # inp2 = np.append(inp, _rnd.binomial(1, bias, (batch_size, 1, element_size + 1)), axis=1)

    # inp = np.append(inp, np.zeros((batch_size, 1, element_size + 1)), axis=1)
    # inp2 = np.append(inp2, np.zeros((batch_size, 1, element_size + 1)), axis=1)
    return inp, inp2


@ex.automain
def run(epochs):
    ntm = build_ntm()
    inp, inp2 = get_inputs()
    with h5py.File(MODEL_WTS, 'r') as f:
        epoch_keys = ['epoch_{:05d}'.format(num) for num in epochs]
        for key in (x for x in epoch_keys if x in f):
            print('In ', key)
            grp = f[key]
            weights = [grp[name] for name in grp if 'Encoder' in name]
            ntm.set_weights(weights)
            ntm_run_data = ntm.get_run_data(inp)
            mem = ntm_run_data['memory'][0, -1, :, :]
            # print(np.transpose(mem))
            ntm_run_data2 = ntm.get_run_data(inp2)
            mem_2 = ntm_run_data2['memory'][0, -1, :, :]
            # print(np.transpose(mem_2))
            length = inp.shape[1] - 1
            mem_flip_roll_2 = np.roll(np.flip(mem_2, axis=0), -length+4, axis=0)
            # print(np.transpose(mem_flip_roll_2 - mem))
            print(np.transpose(mem_flip_roll_2 - mem))
