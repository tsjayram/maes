"""NTM Memory"""
import keras.backend as K
from ntm.tensor_utils import sim, outer_prod

class Memory():
    def __init__(self, mem_t):
        self._memory = mem_t

    def address_read(self, wt):
        return sim(wt, self._memory, adjoint=True)

    def address_add(self, wt, add):
        # memory = memory + sum_{head h} weighted add(h)
        self._memory = self._memory + K.sum(outer_prod(wt, add), axis=-3)
        pass

    def address_erase(self, wt, erase):
        # memory = memory * product_{head h} (1 - weighted erase(h))
        self._memory = self._memory * K.prod(1 - outer_prod(wt, erase), axis=-3)
        pass

    def content_similarity(self, k):
        return sim(k, self._memory, normalize=True)

    @property
    def size(self):
        return K.int_shape(self._memory)

    @property
    def tensor(self):
        return self._memory
