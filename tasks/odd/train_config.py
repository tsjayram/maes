# change below based on task ----
from tasks.odd.build import LOG_ROOT, ex

MEM_FREEZE_TIME = '2018-01-23__08_04_48_PM'
MEM_EPOCH = 20936

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


@ex.config
def mem_weights():
    use_frozen_wts = True
    mem_freeze_wts_file = LOG_ROOT + 'memorize/' + MEM_FREEZE_TIME + '/model_weights.hdf5'
    mem_epoch = MEM_EPOCH

# end change ---


