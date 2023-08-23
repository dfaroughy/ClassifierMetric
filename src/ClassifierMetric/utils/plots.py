import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(train, 
              valid, 
              workdir):
    fig, _ = plt.subplots(figsize=(8,7))
    plt.plot(range(len(train.losses)), np.array(train.losses), color='b', lw=1)
    plt.plot(range(len(valid.losses)), np.array(valid.losses), color='r', lw=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title("loss_min={}, epochs={}".format(round(valid.loss_min,6),len(train.losses)))
    fig.tight_layout()
    plt.savefig(workdir+'/loss.png')
    plt.close()

def plot_class_score(predictions: dict=None,
                     class_labels: dict=None,
                     title: str=None,
                     workdir: str=None, 
                     reference: str=None,
                     figsize=(4,4), 
                     bins=np.arange(-0.03, 1.03, 0.01), 
                     density=False, 
                     xlabel=r'score',
                     ylabel=r'Normalised Entries',
                     xlog=False, 
                     ylog=True,
                     xlim=(0,1),
                     lw=0.75,
                     color=None,
                     alpha=0.2,
                    legend_loc='upper left',
                    legend_fontsize=10,
                    legend=None):

    ref_label = class_labels[reference]
    get_name = {v: k for k, v in class_labels.items()}
    fig, ax = plt.subplots(1, figsize=figsize)    
    N = int(1e10) if density else predictions[-1].shape[0]
    for label, score in predictions.items():
        if label == 'datasets': continue
        test = True if label == -1 else False
        sns.histplot(x=score[:N, ref_label], 
                     bins=bins, 
                     element="step", 
                     log_scale=(xlog, ylog), 
                     lw = 0 if test else lw, 
                     ls = '--' if '(uncond)' in legend[label] else '-',
                     fill=test, 
                     color = color[label],
                     alpha=alpha if test else 1, 
                     ax=ax, 
                     stat='density' if density else 'count',
                     label=get_name[label] if legend is None else legend[label]) 
    plt.xlabel(xlabel)
    plt.ylabel(r'Normalised Entries')
    plt.xlim(xlim)
    if title is not None:
        plt.title('{}'.format(title), fontsize=12)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
    plt.savefig(workdir+'/classifier_score.pdf', bbox_inches='tight', dpi=300)