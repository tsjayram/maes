import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

REPORT_INTERVAL = 100
LOGDIR = '../logs/encode/2018-01-20__10_46_34_AM/'
MAX_TIME = 25000
SKIP_ROWS = 36

train = pd.read_csv(LOGDIR + 'training.log', skiprows=SKIP_ROWS)
train.rename(
    columns={
        'epoch': 'Time',
        'accuracy': 'Training Accuracy',
        'length': 'Length',
        'loss': 'Training Loss',
        'improved': 'Improved'
    }, inplace=True)
train = train.fillna('-')
train = train[train['Improved'].str.contains('\*')]
train = train[train['Time'] <= MAX_TIME]


test = pd.read_csv(LOGDIR + '/test.log', skiprows=SKIP_ROWS)
test.rename(
    columns={
        'epoch': 'Time',
        'accuracy': 'Validation Accuracy',
        'improved': 'Improved'
    }, inplace=True)
test = test.fillna('-')
test = test[test['Improved'].str.contains('\*')]
test = test[test['Time'] <= MAX_TIME]


fig, ax = plt.subplots(figsize=(10, 5))

train.plot(kind='line', style='b-.', y='Loss', x='Time', ax=ax)
train.plot(kind='line', style='r-', y='Accuracy', x='Time', ax=ax)
test.plot(kind='line', style='k:', y='Test Accuracy', x='Time', ax=ax)
ax.legend(loc='best', shadow=True, fontsize='xx-large')
ax.tick_params(labelsize='xx-large')
# ax.xlabel('xlabel', fontsize=24)
# ax.ylabel('ylabel', fontsize=24)
ax.set_xlabel('Iterations', fontsize=18)
fig.savefig(LOGDIR + '/fig_stats.pdf', bbox_inches='tight')
