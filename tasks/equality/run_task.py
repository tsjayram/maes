#!/usr/bin/env python
import os
import arrow
import h5py

# change below based on task ----
from tasks.equality.build import build_ntm, build_data_gen
from tasks.equality.build import ex, TASK_NAME, LOG_ROOT

time_str = '2018-02-04__10_49_30_AM'


@ex.config
def run_config():
    seed = RANDOM_SEED
    N = 80
    batch_size = 32
    length = 64
    bias = 0.5
    num_batches = 100
    epochs = [16926]

# end change ---


LOG_DIR = LOG_ROOT + TASK_NAME + '/' + time_str + '/'
MODEL_WTS = LOG_DIR + 'model_weights.hdf5'
RUNS_DIR = LOG_DIR + 'runs/'
os.makedirs(RUNS_DIR, exist_ok=True)

RANDOM_SEED = 12345


@ex.capture
def make_log(seed, length, N, batch_size, bias):
    format_str = '/seed={}_L={:04d}_N={:04d}_bs={:04d}_bias={:04f}.log'
    logfile = format_str.format(seed, length, N, batch_size, bias)
    return logfile


@ex.capture
def build_run(length):
    ntm = build_ntm()
    data_gen = build_data_gen(ntm, min_len=length, max_len=length)
    return ntm, data_gen


@ex.automain
def run(num_batches, epochs, seed):
    logfile = make_log(seed)
    with open(RUNS_DIR + logfile, 'a') as g:
        time_now = arrow.now().format('YYYY-MM-DD__hh_mm_ss_A')
        g.write(time_now + '\n')
        g.write('batch_num,epoch,batch_acc\n')

    ntm, data_gen = build_run()
    init_state = next(data_gen)
    print(ntm.pretty_print_str())

    with h5py.File(MODEL_WTS, 'r') as f, open(RUNS_DIR + logfile, 'a', 1) as g:
        epoch_keys = ['epoch_{:05d}'.format(num) for num in epochs]

        for n in range(1, num_batches+1):
            inputs, target, length = next(data_gen)
            for key in (x for x in epoch_keys if x in f):
                print('In batch num = {}, {}'.format(n, key))
                grp = f[key]
                weights = [grp[name] for name in grp]
                ntm.set_weights(weights)
                acc = ntm.run(inputs + init_state, target)
                bs = target.shape[0]
                print('Batch size = {}, length = {}: accuracy = {:10.6f}'.format(bs, length, acc))
                g.write('{},{},{:0.6f}\n'.format(n, key, acc))
