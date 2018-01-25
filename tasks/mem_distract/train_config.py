# change below based on task ----
from tasks.mem_distract.build import ex

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


@ex.config
def train_config():
    seed = RANDOM_SEED
    report_interval = REPORT_INTERVAL
    epochs = 200000

    N_train = 256
    train_batch_size = 1
    train_min_num_seq = 6
    train_max_num_seq = 10
    train_avg_len = 7


@ex.config
def test_config():
    N_test = 400
    test_batch_size = 32
    test_min_num_seq = 10
    test_max_num_seq = 15
    test_avg_len = 12

# end change ---


