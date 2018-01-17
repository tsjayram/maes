#!/usr/bin/env python
import logging
import logging.config
import os

import arrow
import h5py
from keras.metrics import binary_crossentropy
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
from sacred import Experiment

from skills.recall.ntm_recall import NTM_Recall
from train.metrics import alt_binary_accuracy
from train.train_ntm import train_ntm

LOG_YAML = 'logger_config.yaml'

LOG_DIR = 'logs/{}'.format(arrow.now().format('YYYY-MM-DD__hh_mm_ss_A'))
os.makedirs(LOG_DIR, exist_ok=True)

LOGFILE = LOG_DIR + '/msgs.log'
TRAIN_LOG = LOG_DIR + '/training.log'
TEST_LOG = LOG_DIR + '/test.log'
MODEL_WTS = LOG_DIR + '/model_weights.hdf5'

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


def logfile():
    return logging.FileHandler(LOGFILE)


with open(LOG_YAML, 'rt') as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('Memory_Recall')

ex = Experiment('train')
ex.logger = logger


@ex.capture
def get_exp_config(_run):
    yaml = YAML()
    yaml.explicit_start = True
    yaml.explicit_end = True
    stream = StringIO()
    yaml.dump(_run.config, stream)
    return stream.getvalue()


@ex.config
def model_config():
    element_size = 8
    tm_state_units = 3
    M = 10
    N_train = 40
    N_test = 128


@ex.config
def train_test_config():
    seed = RANDOM_SEED
    opt_str = 'rmsprop'
    epochs = 50000
    train_batch_size = 1
    test_batch_size = 32
    train_min_len = 3
    train_max_len = 20
    test_len = 64
    report_interval = REPORT_INTERVAL


@ex.config
def mem_weights():
    use_frozen_wts = False
    mem_freeze_wts_file = 'logs/2018-01-01__12_33_40_AM/model_weights.hdf5'
    mem_epoch = 8248


@ex.capture
def build_ntm_train(element_size, tm_state_units, N_train, M, opt_str,
                    train_batch_size, train_min_len, train_max_len, _rnd):
    ntm_train = NTM_Recall(element_size, tm_state_units, N_train, M)
    ntm_train.model.compile(loss=binary_crossentropy, optimizer=opt_str,
                            metrics=[alt_binary_accuracy])
    train_data_gen = ntm_train.data_gen(train_batch_size, train_min_len, train_max_len, _rnd)
    return ntm_train, train_data_gen


@ex.capture
def build_ntm_test(element_size, tm_state_units, N_test, M,
                   test_batch_size, test_len, _rnd):
    ntm_test = NTM_Recall(element_size, tm_state_units, N_test, M)
    test_data_gen = ntm_test.data_gen(test_batch_size, test_len, test_len, _rnd)
    return ntm_test, test_data_gen


@ex.capture
def mem_load_weights(ntm, use_frozen_wts, mem_freeze_wts_file, mem_epoch):
    if use_frozen_wts:
        layer = ntm.model.get_layer('NTM_Layer_Memorize')
        epoch_key = 'epoch_{:05d}'.format(mem_epoch)
        with h5py.File(mem_freeze_wts_file, 'r') as f:
            print('In ', epoch_key)
            grp = f[epoch_key]
            weights = [grp[name] for name in grp if 'Memorize' in name]
            layer.set_weights(weights)
            layer.trainable = False


@ex.automain
def main(epochs, report_interval, _log, seed):
    _log.info('Seed = {}'.format(seed))
    exp_config_str = get_exp_config()
    _log.info('\n' + exp_config_str)
    ntm_train, train_data_gen = build_ntm_train()
    ntm_test, test_data_gen = build_ntm_test()

    mem_load_weights(ntm_train)

    _log.info(ntm_train.pretty_print_str())
    print(ntm_train.pretty_print_str())
    input('pause')

    with open(TRAIN_LOG, 'w', 1) as train_file, \
            open(TEST_LOG, 'w', 1) as test_file, \
            h5py.File(MODEL_WTS, 'a') as model_wts_file:
        train_file.write(exp_config_str)
        test_file.write(exp_config_str)
        train_ntm(ntm_train, train_data_gen, train_file,
                  ntm_test, test_data_gen, test_file, model_wts_file,
                  epochs, report_interval, _log)
