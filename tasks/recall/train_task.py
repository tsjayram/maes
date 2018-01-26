#!/usr/bin/env python
import logging
import logging.config
import os
import numpy as np
import arrow
import h5py
from ruamel.yaml import YAML

from model.train import train_ntm
from tasks.utils import get_exp_config, pause

# change below based on task ----
from tasks.recall.build import build_ntm, build_data_gen
from tasks.recall.build import ex, TASK_NAME, LOG_ROOT
from tasks.recall.train_config import get_train_status
# end change ---

time_str = arrow.now().format('YYYY-MM-DD__hh_mm_ss_A')
LOG_DIR = LOG_ROOT + TASK_NAME + '/' + time_str + '/'
os.makedirs(LOG_DIR, exist_ok=True)
LOGFILE = LOG_DIR + 'msgs.log'
TRAIN_LOG = LOG_DIR + 'training.log'
TEST_LOG = LOG_DIR + 'test.log'
MODEL_WTS = LOG_DIR + 'model_weights.hdf5'
LOG_YAML = 'logger_config.yaml'


def logfile():
    return logging.FileHandler(LOGFILE)


with open(LOG_YAML, 'rt') as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(TASK_NAME)
ex.logger = logger


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
def build_train(N_train, train_batch_size, train_min_len, train_max_len):
    ntm = build_ntm(N=N_train)
    mem_freeze(ntm)
    data_gen = build_data_gen(ntm, train_batch_size, train_min_len, train_max_len)
    return ntm, data_gen


@ex.capture
def build_test(N_test, test_batch_size, test_len):
    ntm = build_ntm(N=N_test)
    data_gen = build_data_gen(ntm, test_batch_size, test_len, test_len)
    return ntm, data_gen


@ex.automain
def main(epochs, _log, seed, _run):
    _log.info('Seed = {}'.format(seed))

    exp_config_str = get_exp_config(_run)
    _log.info('\n' + exp_config_str)

    ntm_train, train_data_gen = build_train()
    ntm_test, test_data_gen = build_test()
    _log.info(ntm_train.pretty_print_str())

    train_status = get_train_status()
    next(train_status)

    with open(TRAIN_LOG, 'w', 1) as train_file, \
            open(TEST_LOG, 'w', 1) as test_file, \
            h5py.File(MODEL_WTS, 'a') as model_wts_file:
        train_file.write(exp_config_str)
        test_file.write(exp_config_str)
        pause()
        train_ntm(ntm_train, train_data_gen, train_file,
                  ntm_test, test_data_gen, test_file, model_wts_file,
                  epochs, train_status, _log)
