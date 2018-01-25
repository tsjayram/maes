# change below based on task ----
from tasks.memorize.build import ex

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
    report_interval = REPORT_INTERVAL

# end change ---


