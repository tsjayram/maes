import sys

import numpy as np
from keras.metrics import binary_crossentropy

from model.metrics import alt_binary_accuracy, alt_binary_cross_entropy

np.set_printoptions(threshold=np.nan)


def train_ntm(ntm_train, train_data_gen, train_file,
              ntm_test, test_data_gen, test_file, model_wts_file,
              epochs, train_status, log):

    ntm_train.model.compile(loss=binary_crossentropy, optimizer='rmsprop',
                            metrics=[alt_binary_accuracy])

    train_file.write('\nepoch,accuracy,length,loss,improved\n')
    train_init_state = next(train_data_gen)

    test_file.write('\nepoch,accuracy,improved\n')
    test_init_state = next(test_data_gen)
    test_inputs, test_target, test_length = next(test_data_gen)
    format_str = 'Test Shape:\n  Output = {}, Length = {}'
    log.debug(format_str.format(test_target.shape, test_length))

    for epoch in range(1, epochs+1):
        log.info('Starting epoch {}'.format(epoch))
        flag_ok, model_wt_str = ntm_train.params_str()
        if flag_ok:
            log.info('Epoch {} weights ok'.format(epoch))
        else:
            log.warning('Epoch {} Nan in weights'.format(epoch))
            log.debug(model_wt_str)
            sys.exit('Exiting because of nan in training')

        train_inputs, train_target, train_length = next(train_data_gen)
        loss, acc = ntm_train.model.train_on_batch(train_inputs + train_init_state, train_target)
        format_str = '\n==>Stats at epoch {:05d}:\n'
        format_str = format_str + '    acc={:12.10f}; loss={:12.10f}; length={:4d}'
        log.info(format_str.format(epoch, acc, loss, train_length))
        format_str = '{:05d}, {:12.10f}, {:4d}, {:12.10f}'
        train_file.write(format_str.format(epoch, acc, train_length, loss))

        args = (epoch, train_length, loss)
        test_flag, improved_loss = train_status.send(args)

        if improved_loss:
            train_file.write(', *\n')
        else:
            train_file.write('\n')

        if not test_flag:
            continue

        # testing
        log.info('Testing accuracy in epoch {}'.format(epoch))
        weights = ntm_train.get_weights()
        ntm_test.set_weights(weights)

        group = model_wts_file.create_group('epoch_{:05d}'.format(epoch))
        for _, name, weight in ntm_test.params_gen():
            group.create_dataset(name, data=weight)

        acc = ntm_test.run(test_inputs + test_init_state, test_target)
        if acc == -1:
            log.warning('Test Nan encountered.')
            _, model_wt_str = ntm_test.params_str()
            log.debug(model_wt_str)
            sys.exit('Exiting because of nan in test')

        log.info('\n==>Accuracy with length {} was: {:12.10f}'.format(test_length, acc))
        test_file.write('{:05d}, {:12.10f}'.format(epoch, acc))

        if improved_loss:
            test_file.write(', *\n')
        else:
            test_file.write('\n')
