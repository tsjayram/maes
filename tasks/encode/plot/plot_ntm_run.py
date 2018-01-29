import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def initialize_plot_area():
    plt.ioff()
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(50, 100)

    axes = {
        'input': plt.subplot(gs[0:10, 5:95]),
        'read': plt.subplot(gs[15:50, 0:35]),
        'memory': plt.subplot(gs[15:50, 40:60]),
        'write': plt.subplot(gs[15:50, 65:100]),
    }

    axes['input'].set_title('Input', loc='center')
    axes['memory'].set_title('Memory', loc='center')
    axes['read'].set_title('Read Attention', loc='center')
    axes['write'].set_title('Write Attention', loc='center')

    axes['input'].set_ylabel('Word')
    axes['input'].set_yticklabels([])

    axes['input'].set_xlabel('Sequence Index')

    axes['memory'].set_xlabel('Slots')

    axes['read'].set_ylabel('Memory Address')
    for label in ['read', 'write']:
        axes[label].set_xlabel('Time')

    for key in ['input','read', 'write', 'memory']:
        axes[key].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[key].yaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, axes, gs


def plot_ntm_run(inp, ntm_run_data):

    # use first sample
    index = 0

    fig, axes, gs = initialize_plot_area()
    r_head_idx = 0
    w_head_idx = 0

    plot_data = {
        'input': np.transpose(inp[index]),
        'read': np.transpose(ntm_run_data['read'][index, :, r_head_idx, :]),
        'write': np.transpose(ntm_run_data['write'][index, :, w_head_idx, :]),
    }

    for key, array in plot_data.items():
        axes[key].imshow(array, cmap='magma', aspect='auto')

    mem_array = ntm_run_data['memory'][index, -1, :, :]
    obj = axes['memory'].imshow(mem_array, cmap='magma', aspect='auto')
    fig.colorbar(obj, ax=axes['memory'])

    for key in axes:
        axes[key].set_adjustable('box-forced')
        axes[key].autoscale(False)

    gs.tight_layout(fig)
    return fig

