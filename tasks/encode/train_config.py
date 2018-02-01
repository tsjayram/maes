from tasks.utils import train_status_gen

# change below based on task ----
from tasks.encode.build import ex
from tasks.encode.build import build_ntm, build_data_gen

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


@ex.config
def train_test_config():
    seed = RANDOM_SEED
    epochs = 50000
    N_train = 30
    N_test = 80
    train_batch_size = 1
    test_batch_size = 64
    train_min_len = 3
    train_max_len = 20
    test_len = 64
    bias = 0.5
    report_interval = REPORT_INTERVAL

# end change ---


@ex.capture
def get_train_status(train_max_len, report_interval):
    threshold = (train_max_len * 3) // 4
    return train_status_gen(threshold, report_interval)


@ex.capture
def build_train(N_train, train_batch_size, train_min_len, train_max_len, bias):
    ntm = build_ntm(N=N_train)
    data_gen = build_data_gen(ntm, train_batch_size, train_min_len, train_max_len, bias)
    return ntm, data_gen


@ex.capture
def build_test(N_test, test_batch_size, test_len, bias):
    ntm = build_ntm(N=N_test)
    data_gen = build_data_gen(ntm, test_batch_size, test_len, test_len, bias)
    return ntm, data_gen
