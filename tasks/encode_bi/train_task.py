#!/usr/bin/env python
import logging
import logging.config
import os
import arrow
import h5py
from ruamel.yaml import YAML

from model.train import train_ntm
from tasks.utils import get_exp_config, pause

# change below based on task ----
from tasks.encode_bi.build import ex, TASK_NAME, LOG_ROOT
from tasks.encode_bi.train_config import build_train, build_test, get_train_status
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
