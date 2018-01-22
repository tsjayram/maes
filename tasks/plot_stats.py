import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

REPORT_INTERVAL = 100
LOGDIR = 'logs/reverse/2018-01-18__02_09_53_AM'
MAX_TIME = 50000
SKIP_ROWS = 22

train = pd.read_csv(LOGDIR + '/training.log', skiprows=SKIP_ROWS)
train.rename(
    columns={
        'epoch': 'Time',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
    }, inplace=True)

train['Accuracy'] = train['Accuracy'].rolling(REPORT_INTERVAL, center=True, min_periods=1).mean()
train['Loss'] = train['Loss'].rolling(REPORT_INTERVAL, center=True, min_periods=1).mean()
train = train[(train['Time'] <= MAX_TIME) & (train['Time'] % 100 == 0)]

test = pd.read_csv(LOGDIR + '/test.log', skiprows=SKIP_ROWS)
test.rename(
    columns={
        'epoch': 'Time',
        'accuracy': 'Test Accuracy',
    }, inplace=True)
test = test[test['Time'] <= MAX_TIME]

fig, ax = plt.subplots()

train.plot(kind='line', y='Loss', x='Time', ax=ax)
train.plot(kind='line', y='Accuracy', x='Time', ax=ax)
test.plot(kind='line', y='Test Accuracy', x='Time', ax=ax)
ax.legend(loc='best', shadow=True, fontsize='small')
fig.savefig(LOGDIR + '/fig_stats.pdf', bbox_inches='tight')
