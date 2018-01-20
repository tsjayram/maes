#!/usr/bin/env python
import os

import h5py
from sacred import Experiment
from skills.odd.odd import NTM_Solve


LOG_DIR = 'permanent_logs/2018-01-16__12_04_47_PM'
MODEL_WTS = LOG_DIR + '/model_weights.hdf5'
RUNS_DIR = LOG_DIR + '/runs'
os.makedirs(RUNS_DIR, exist_ok=True)

RANDOM_SEED = 12345

ex = Experiment('runs')


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    N = 128
    M = 10


@ex.config
def run_config():
    seed = RANDOM_SEED
    length = 64
    batch_size = 32
    all_epochs = False
    epoch_min = 12000
    epoch_max = 13500


@ex.capture
def make_log(seed, length, N):
    logfile = '/seed={}_L={:04d}_N={:04d}.log'.format(seed, length, N)
    return logfile


@ex.capture
def build_ntm(element_size, tm_state_units, N, M):
    return NTM_Solve(element_size, tm_state_units, N, M)


@ex.capture
def data_gen(ntm, batch_size, length, _rnd):
    return ntm.data_gen(batch_size, length, length, _rnd)


@ex.automain
def run(all_epochs, epoch_min, epoch_max, seed):
    logfile = make_log(seed)
    with open(RUNS_DIR + logfile, 'a') as g:
        g.write('epoch,batch_acc\n')

    ntm = build_ntm()
    print(ntm.pretty_print_str())

    gen = data_gen(ntm)
    inputs, init_state, target, length = next(gen)

    with h5py.File(MODEL_WTS, 'r') as f, open(RUNS_DIR + logfile, 'a', 1) as g:
        if all_epochs:
            epoch_keys = [epoch_str for epoch_str in f]
        else:
            epoch_keys = ['epoch_{:05d}'.format(num) for num in range(epoch_min, epoch_max+1)]

        for key in (x for x in epoch_keys if x in f):
            print('In ', key)
            grp = f[key]
            weights = [grp[name] for name in grp]
            ntm.set_weights(weights)
            acc = ntm.run(inputs + init_state, target)
            bs = target.shape[0]
            print('Batch size = {}, length = {}: accuracy = {:10.6f}'.format(bs, length, acc))
            g.write('{},{:0.6f}\n'.format(key, acc))
