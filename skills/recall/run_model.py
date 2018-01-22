#!/usr/bin/env python
import os

import h5py

from skills.recall.build_tools import ex, LOG_ROOT
from skills.recall.build_tools import build_ntm, build_data_gen

LOG_DIR = LOG_ROOT + 'recall/2018-01-20__10_46_34_AM/'
MODEL_WTS = LOG_DIR + 'model_weights.hdf5'
RUNS_DIR = LOG_DIR + 'runs/'
os.makedirs(RUNS_DIR, exist_ok=True)

RANDOM_SEED = 12345


@ex.config
def run_config():
    seed = RANDOM_SEED
    N = 128
    batch_size = 32
    length = 64
    all_epochs = False
    epoch_min = 11400
    epoch_max = 22500


@ex.capture
def make_log(seed, length, N):
    logfile = '/seed={}_L={:04d}_N={:04d}.log'.format(seed, length, N)
    return logfile


@ex.capture
def build_run(length):
    ntm = build_ntm()
    data_gen = build_data_gen(ntm, min_len=length, max_len=length)
    return ntm, data_gen


@ex.automain
def run(all_epochs, epoch_min, epoch_max, seed):
    logfile = make_log(seed)
    with open(RUNS_DIR + logfile, 'a') as g:
        g.write('epoch,batch_acc\n')

    ntm, data_gen = build_run()
    print(ntm.pretty_print_str())

    inputs, init_state, target, length = next(data_gen)

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
