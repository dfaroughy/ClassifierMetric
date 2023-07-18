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

def plot_class_score(test_probs, 
                     model_probs, 
                     workdir, 
                     label,
                     figsize=(4,4), 
                     bins=np.arange(-0.03, 1.03, 0.01), 
                     xlog=False, 
                     ylog=True, 
                     xlim=(0,1), 
                     legends=None):

    fig, ax = plt.subplots(1, figsize=figsize)
    sns.histplot(x=test_probs[..., label], 
                 color='k', 
                 bins=bins, 
                 element="step", 
                 log_scale=(xlog, ylog), 
                 lw=0, 
                 fill=True, 
                 alpha=0.2, 
                 ax=ax, 
                 label='test')

    for i, x in enumerate(model_probs):
        x = x[..., label] 
        legends = [None] * len(model_probs) if legends is None else legends
        sns.histplot(x=x, 
                     bins=bins, 
                     element="step", 
                     log_scale=(xlog, ylog), 
                     lw=0.75, 
                     fill=False, 
                     alpha=1, 
                     ax=ax, 
                     label=legends[i]) 
    plt.xlabel(r'score')
    plt.xlim(xlim)
    plt.title(r'Classifier, score_label={}'.format(label))
    plt.legend(loc='upper left')
    plt.savefig(workdir+'/classifier_score.png')
