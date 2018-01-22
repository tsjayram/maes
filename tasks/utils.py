import inspect

import arrow
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO


def get_exp_config(_run):
    yaml = YAML()
    stream = StringIO()
    start_time = arrow.get(_run.start_time, 'UTC').to('US/Pacific')
    time_dict = {'Start time': start_time.format('YYYY-MM-DD hh:mm:ss A')}
    yaml.dump(time_dict, stream)
    yaml.explicit_start = True
    yaml.explicit_end = True
    yaml.dump(_run.experiment_info, stream)
    print(_run.experiment_info)
    yaml.dump(_run.config, stream)
    return stream.getvalue()


def train_status_gen(threshold, report_interval):
    best_loss = 1.0
    test_flag = False
    improved_loss = False
    while True:
        args = (yield test_flag, improved_loss)
        epoch, train_length, loss = args
        if (loss < best_loss) and (train_length > threshold):
            improved_loss = True
            best_loss = loss
        else:
            improved_loss = False

        if (epoch % report_interval == 0) or improved_loss:
            test_flag = True
        else:
            test_flag = False


def pause():
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    format_str = 'Pause at line {} in function {} of file {}'
    input(format_str.format(info.lineno, info.function, info.filename))
