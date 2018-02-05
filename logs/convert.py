#!/usr/bin/env python
import h5py

LOG_DIR = './encode/2018-01-20__10_46_34_AM/'
FILE_MEM = LOG_DIR + 'model_weights.hdf5'
FILE_ENC = LOG_DIR + 'model_weights_new.hdf5'

with h5py.File(FILE_MEM, 'r') as file_mem, h5py.File(FILE_ENC, 'w') as file_enc:
    for epoch_key in file_mem:
        group_mem = file_mem[epoch_key]
        group_enc = file_enc.create_group(epoch_key)
        for name_mem in group_mem:
            weights = group_mem[name_mem]
            if 'Memorize' in name_mem:
                name_enc = name_mem.replace('Memorize', 'Encoder')
            elif 'Solve' in name_mem:
                name_enc = name_mem.replace('Solve', 'Solver')
            else:
                name_enc = name_mem
            group_enc.create_dataset(name_enc, data=weights)
