import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def initialize_plot_area():
    plt.ioff()
    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(70, 80)

    axes = {
        'input': plt.subplot(gs[0:15, :80]),
        'read': plt.subplot(gs[30:50, 0:30]),
        'write': plt.subplot(gs[50:70, 0:30]),
        'memory': plt.subplot(gs[30:50, 32:80]),
    }

    axes['input'].set_title('Input', loc='right')
    axes['memory'].set_title('Memory', loc='right')
    axes['read'].set_title('Read Attention', loc='right')
    axes['write'].set_title('Write Attention', loc='right')

    axes['input'].set_ylabel('Element')
    axes['input'].set_yticklabels([])

    axes['input'].set_xlabel('Sequence')

    axes['read'].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes['write'].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes['memory'].set_xlabel('Memory Location')
    axes['memory'].set_ylabel('Content')

    axes['read'].set_xlabel('Time')
    for label in ['read', 'write']:
        axes[label].set_ylabel('Memory Location')
        # axes[label].yaxis.set_label_position('right')
        # axes[label].yaxis.set_ticks_position('right')
        axes[label].tick_params(axis='y', which='both', labelleft='on', labelright='off')

    return fig, axes, gs


def plot_ntm_run(seq, ntm_run_data):
    fig, axes, gs = initialize_plot_area()
    r_head_idx = 0
    w_head_idx = 0

    plot_data = {
        'input': np.transpose(seq),
        'read': np.transpose(ntm_run_data['read'][:, r_head_idx, :]),
        'write': np.transpose(ntm_run_data['write'][:, w_head_idx, :]),
    }

    for key, array in plot_data.items():
        axes[key].imshow(array, cmap='magma', aspect='auto')

    mem_array = np.transpose(ntm_run_data['memory'][-1])
    obj = axes['memory'].imshow(mem_array, cmap='magma', aspect='auto')
    fig.colorbar(obj, ax=axes['memory'])

    for key in axes:
        axes[key].set_adjustable('box-forced')
        axes[key].autoscale(False)

    gs.tight_layout(fig)
    # plt.pause(0.001)
    return fig

