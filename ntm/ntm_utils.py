import io
import numpy as np

from keras.utils import print_summary
from keras import backend as K

np.set_printoptions(threshold=np.nan)


def params_gen(model):
    for i, (tensor, weight) in enumerate(zip(model.trainable_weights, model.get_weights())):
        # name hack relies on tensorflow backend
        tokens = tensor.name.split('/')
        tokens = ['{:02d}'.format(i + 1)] + tokens  # add index
        name = '___'.join(tokens)
        yield tensor, name, weight


def params_str(model):
    h_line = '*' * 80 + '\n'
    flag_ok = True
    out = io.StringIO()
    for tensor, name, weight in params_gen(model):
        out.write(h_line + '\n')
        out.write('Name: {} >>>> Shape: {}\n\n'.format(name, K.int_shape(tensor)))
        out.write('Weights:\n {}\n'.format(weight))
        out.write('\n' + h_line + '\n')
        if np.isnan(np.sum(weight)):
            flag_ok = False
    return flag_ok, out


def pretty_print(model):
    def print_fn(x):
        out.write(x)
        out.write('\n')

    h_line = '*' * 80 + '\n'
    out = io.StringIO()
    out.write('\n' + h_line)
    print_summary(model, line_length=200, positions=[.25, .7, .8, 1.],
                  print_fn=print_fn)
    out.write(h_line)
    return out.getvalue()
