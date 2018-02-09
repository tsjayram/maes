import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

REPORT_INTERVAL = 100
LOGDIR = '../logs/n_back/2018-02-06__09_15_14_AM/'
MAX_TIME = 50000
SKIP_ROWS = 40

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


fig, ax = plt.subplots()

train.plot(kind='line', y='Training Loss', x='Time', ax=ax)
train.plot(kind='line', y='Training Accuracy', x='Time', ax=ax)
test.plot(kind='line', y='Validation Accuracy', x='Time', ax=ax)
ax.legend(loc='best', shadow=True, fontsize='small')
fig.savefig(LOGDIR + '/fig_stats.pdf', bbox_inches='tight')
