import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def initialize_plot_area():
    plt.ioff()
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(35, 80)

    axes = {
        'input': plt.subplot(gs[0:35, 0:15]),
        'write': plt.subplot(gs[0:35, 20:55]),
        'memory': plt.subplot(gs[0:35, 60:80]),
    }

    axes['input'].set_title('Input', loc='center')
    axes['memory'].set_title('Memory', loc='center')
    axes['write'].set_title('Write Attention', loc='center')

    axes['input'].set_xlabel('Word')
    axes['input'].set_xticklabels([])
    axes['input'].set_ylabel('Sequence Index')

    axes['memory'].set_xlabel('Memory Word Position')
    axes['memory'].set_ylabel('Address')

    axes['write'].set_xlabel('Time')
    axes['write'].set_ylabel('Attention')

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
    axes['input'].imshow(inp_array, cmap='Greys', aspect='auto')

    write_array = ntm_run_data['write'][index, :, w_head_idx, :]
    axes['write'].imshow(np.transpose(write_array), cmap='Greys', aspect='auto')

    mem_array = ntm_run_data['memory'][index, -1, :, :]
    obj = axes['memory'].imshow(mem_array, cmap='Greys', aspect='auto')
    fig.colorbar(obj, ax=axes['memory'])

    for key in axes:
        axes[key].set_adjustable('box-forced')
        axes[key].autoscale(False)

    gs.tight_layout(fig)
    return fig

