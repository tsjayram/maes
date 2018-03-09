import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def initialize_plot_area():
    plt.ioff()
    fig = plt.figure(figsize=(18, 9.6))
    gs = GridSpec(32, 60)

    axes = {
        'write': plt.subplot(gs[0:20, 0:35]),
        'input': plt.subplot(gs[22:32, 0:35]),
        'memory': plt.subplot(gs[0:32, 37:60]),
    }

    axes['input'].set_title('Input', loc='center', fontsize=22)
    axes['memory'].set_title('Memory', loc='center', fontsize=22)
    axes['write'].set_title('Write Head', loc='center', fontsize=22)

    axes['input'].set_xlabel('Sequence Index', fontsize=20)
    axes['input'].set_yticklabels([])
    axes['input'].set_ylabel('Word', fontsize=20)
    axes['input'].tick_params(labelsize='xx-large')

    axes['memory'].set_xlabel('Vector Coordinates', fontsize=20)
    axes['memory'].set_ylabel('Address', fontsize=20)
    axes['memory'].tick_params(labelsize='xx-large')

    axes['write'].set_xlabel('Time', fontsize=20)
    axes['write'].set_ylabel('Soft Attention', fontsize=20)
    axes['write'].tick_params(labelsize='xx-large')

    for key in ['input', 'write', 'memory']:
        axes[key].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[key].yaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, axes, gs


def plot_ntm_run(inp, ntm_run_data):

    # use first sample
    index = 0

    fig, axes, gs = initialize_plot_area()
    w_head_idx = 0

    inp_array = inp[index, 1:, 1:]
    axes['input'].imshow(np.transpose(inp_array), cmap='magma', aspect='auto')

    write_array = ntm_run_data['write'][index, 1:, w_head_idx, :]
    axes['write'].imshow(np.transpose(write_array), cmap='magma', aspect='auto')

    mem_array = ntm_run_data['memory'][index, -1, :, :]
    obj = axes['memory'].imshow(mem_array, cmap='magma', aspect='auto')
    cbar = fig.colorbar(obj, ax=axes['memory'])
    cbar.ax.tick_params(labelsize=18)

    for key in axes:
        axes[key].set_adjustable('box-forced')
        axes[key].autoscale(False)

    gs.tight_layout(fig)
    return fig

