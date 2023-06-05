from matplotlib import pyplot as plt

SMALL_SIZE = 11
MEDIUM_SIZE = 11
BIGGER_SIZE = 14

plt.style.use('science')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('figure', figsize=(4, 3))
# plt.rcParams.update({
#     # "font.size": 12,
#     # "figure.figsize": [6.4, 4.8] # default
# })

plt.rc('xtick.minor', visible=False)
plt.rc('ytick.minor', visible=False)

def scale_fig(w, h, scale_factor):
   return w*scale_factor, h*scale_factor


def label_fig(fig, x_label=None, y_label=None, x_offset=.04, y_offset=.04):
    if x_label is not None:
        fig.text(0.5, x_offset, x_label, ha='center', va='center')
    if y_label is not None:
        fig.text(y_offset, 0.5, y_label, ha='center', va='center', rotation='vertical')
    # fig.supxlabel(x_label)
    # fig.supylabel(y_label)

    
def hline(ax, y, color='tab:grey', linestyle='--', label=None):
    ax.axhline(y=y, color=color, linestyle=linestyle, label=label)


def point(ax, x, y, color='red', marker='o', label=None):
    ax.plot([x], [y], marker=marker, markersize=4, color=color, linestyle='none', label=label)

    
def row_fig(n, sharey=True, figsize=(4, 3)):
    fig, axs = plt.subplots(
            1, n,
            sharey=sharey,
            # subplot_kw={'adjustable':'box', 'aspect':'equal'},
            figsize=figsize
        )
    if sharey:
        fig.subplots_adjust(wspace=.1)
    # fig.tight_layout(rect=[.03, .03, 1, .95])
    return fig, axs


def ax_plot(ax, x, ys, labels, colors=None):
    if not colors:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(ys)]
    for y, l, c in zip(ys, labels, colors):
        ax.plot(x, y, '-', label=l, color=c)
    ax.legend()
        
def re_part_labels(indices):
    labels = list()
    for i in indices:
        labels.append(f'Re$(\lambda_{i})$')
    return labels


def im_part_labels(indices):
    labels = list()
    for i in indices:
        labels.append(f'Im$(\lambda_{i})$')
    return labels


def save_to_img_dir(fig, file_name):
    import os
    if '.pdf' in file_name:
        f = file_name
    else:
        f = f'{file_name}.pdf'
    p = os.path.join('/Users/daniel/dev/phd/dde_latex/img', f'{file_name}.pdf')
    fig.savefig(p)
