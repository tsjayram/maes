#!/usr/bin/env python
import os
import logging
import logging.config

import arrow
import h5py
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

from skills.recall.build_tools import ex, LOG_ROOT, pause
from skills.recall.build_tools import build_ntm, mem_freeze, build_data_gen

from skills.train import train_ntm

time_now = arrow.now().format('YYYY-MM-DD__hh_mm_ss_A')
LOG_DIR = LOG_ROOT + 'recall/{}/'.format(time_now)
os.makedirs(LOG_DIR, exist_ok=True)


LOGFILE = LOG_DIR + '/msgs.log'
TRAIN_LOG = LOG_DIR + '/training.log'
TEST_LOG = LOG_DIR + '/test.log'
MODEL_WTS = LOG_DIR + '/model_weights.hdf5'
LOG_YAML = 'logger_config.yaml'

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


@ex.config
def train_test_config():
    seed = RANDOM_SEED
    epochs = 50000
    N_train = 40
    N_test = 128
    train_batch_size = 1
    test_batch_size = 32
    train_min_len = 3
    train_max_len = 20
    test_len = 64
    report_interval = REPORT_INTERVAL


def logfile():
    return logging.FileHandler(LOGFILE)


with open(LOG_YAML, 'rt') as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('Memory_Recall')
ex.logger = logger


@ex.capture
def get_exp_config(_run):
    yaml = YAML()
    yaml.explicit_start = True
    yaml.explicit_end = True
    stream = StringIO()
    yaml.dump(_run.start_time, stream)
    yaml.dump(_run.experiment_info, stream)
    yaml.dump(_run.config, stream)
    return stream.getvalue()


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
def main(epochs, report_interval, _log, seed):
    _log.info('Seed = {}'.format(seed))
    exp_config_str = get_exp_config()
    _log.info('\n' + exp_config_str)

    ntm_train, train_data_gen = build_train()
    ntm_test, test_data_gen = build_test()
    _log.info(ntm_train.pretty_print_str())

    with open(TRAIN_LOG, 'w', 1) as train_file, \
            open(TEST_LOG, 'w', 1) as test_file, \
            h5py.File(MODEL_WTS, 'a') as model_wts_file:
        train_file.write(exp_config_str)
        test_file.write(exp_config_str)
        pause()
        train_ntm(ntm_train, train_data_gen, train_file,
                  ntm_test, test_data_gen, test_file, model_wts_file,
                  epochs, report_interval, _log)
