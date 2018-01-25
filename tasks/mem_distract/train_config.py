# change below based on task ----
from tasks.mem_distract.build import ex

RANDOM_SEED = 12345
REPORT_INTERVAL = 100


@ex.config
def train_config():
    seed = RANDOM_SEED
    report_interval = REPORT_INTERVAL
    epochs = 200000

    N_train = 60
    train_batch_size = 1
    train_min_num_seq = 2
    train_max_num_seq = 6
    train_avg_len = 5


@ex.config
def test_config():
    N_test = 180
    test_batch_size = 32 
    test_min_num_seq = 6
    test_max_num_seq = 8
    test_avg_len = 10

# end change ---
