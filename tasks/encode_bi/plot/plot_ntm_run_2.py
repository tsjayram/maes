import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def initialize_plot_area():
    plt.ioff()
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(35, 60)

    axes = {
        'input': plt.subplot(gs[25:35, 0:35]),
        'write': plt.subplot(gs[0:20, 0:35]),
        'memory': plt.subplot(gs[0:20, 40:60]),
    }

    axes['input'].set_title('Input', loc='center')
    axes['memory'].set_title('Memory', loc='center')
    axes['write'].set_title('Write Attention', loc='center')

    axes['input'].set_xlabel('Sequence Index')
    axes['input'].set_yticklabels([])
    axes['input'].set_ylabel('Word')

    axes['memory'].set_xlabel('Memory Word Position', fontsize=18)
    axes['memory'].set_ylabel('Address', fontsize=18)
    axes['memory'].tick_params(labelsize='xx-large')

    axes['write'].set_xlabel('Time')
    axes['write'].set_ylabel('Soft Attention')

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
    axes['input'].imshow(np.transpose(inp_array), cmap='Greys', aspect='auto')

    write_array = ntm_run_data['write'][index, 1:, w_head_idx, :]
    axes['write'].imshow(np.transpose(write_array), cmap='Greys', aspect='auto')

    mem_array = ntm_run_data['memory'][index, -1, :, :]
    obj = axes['memory'].imshow(mem_array, cmap='Greys', aspect='auto')
    fig.colorbar(obj, ax=axes['memory'])

    for key in axes:
        axes[key].set_adjustable('box-forced')
        axes[key].autoscale(False)

    gs.tight_layout(fig)
    return fig

