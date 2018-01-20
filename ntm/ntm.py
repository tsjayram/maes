from abc import ABC, abstractmethod
import io
import numpy as np

from keras.utils import print_summary
from keras import backend as K


class NTM(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def _build_model(self):
        pass

    @property
    def model(self):
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def run(self, inputs, target):
        batch_size = target.shape[0]
        out = self.model.predict(inputs, batch_size=batch_size)
        if np.isnan(out.sum()):  # checks for a NaN anywhere
            acc = -1
            return acc

        predict = np.round(out)
        acc = 1 - np.abs(predict - target)
        acc = acc.mean()
        return acc

    def params_gen(self):
        model = self.model
        n = 1
        for layer in model.layers:
            for tensor, weight in zip(layer.weights, layer.get_weights()):
                # name hack relies on tensorflow backend
                tokens = tensor.name.split('/')
                tokens = ['{:02d}'.format(n)] + tokens  # add index
                name = '___'.join(tokens)
                n = n + 1
                yield tensor, name, weight

    def params_str(self):
        h_line = '*' * 80 + '\n'
        flag_ok = True
        out = io.StringIO()
        for tensor, name, weight in self.params_gen():
            out.write(h_line + '\n')
            out.write('Name: {} >>>> Shape: {}\n\n'.format(name, K.int_shape(tensor)))
            out.write('Weights:\n {}\n'.format(weight))
            out.write('\n' + h_line + '\n')
            if np.isnan(np.sum(weight)):
                flag_ok = False
        return flag_ok, out

    def pretty_print_str(self):
        model = self.model

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
