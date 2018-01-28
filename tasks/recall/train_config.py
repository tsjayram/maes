import numpy as np
import h5py
from tasks.utils import train_status_gen

# change below based on task ----
from tasks.recall.build import ex, LOG_ROOT
from tasks.recall.build import build_ntm, build_data_gen

ENCODER_FREEZE_TIME = '2018-01-28__12_40_36_AM'
ENCODER_EPOCH = 27987

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


@ex.config
def train_test_config():
    seed = RANDOM_SEED
    epochs = 50000
    N_train = 40
    N_test = 128
    train_batch_size = 1
    test_batch_size = 64
    train_min_len = 3
    train_max_len = 20
    test_len = 64
    bias = 0.5
    report_interval = REPORT_INTERVAL


@ex.capture
def get_train_status(train_max_len, report_interval):
    threshold = (train_max_len * 3) // 4
    return train_status_gen(threshold, report_interval)

# end change ---


@ex.config
def encoder_weights():
    use_frozen_wts = True
    encoder_freeze_wts_file = LOG_ROOT + 'encode/' + ENCODER_FREEZE_TIME + '/model_weights.hdf5'
    encoder_epoch = ENCODER_EPOCH


@ex.capture
def freeze_encoder(ntm, use_frozen_wts, encoder_freeze_wts_file, encoder_epoch):
    if use_frozen_wts:
        epoch_key = 'epoch_{:05d}'.format(encoder_epoch)
        with h5py.File(encoder_freeze_wts_file, 'r') as f:
            print('In ', epoch_key)
            grp = f[epoch_key]
            weights = [np.array(grp[name]) for name in grp if 'Encoder' in name]
            ntm.encoder_freeze_weights(weights)


@ex.capture
def build_train(N_train, train_batch_size, train_min_len, train_max_len, bias):
    ntm = build_ntm(N=N_train)
    freeze_encoder(ntm)
    data_gen = build_data_gen(ntm, train_batch_size, train_min_len, train_max_len, bias)
    return ntm, data_gen


@ex.capture
def build_test(N_test, test_batch_size, test_len, bias):
    ntm = build_ntm(N=N_test)
    data_gen = build_data_gen(ntm, test_batch_size, test_len, test_len, bias)
    return ntm, data_gen
