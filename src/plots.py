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
                     workdir: str=None, 
                     reference: str=None,
                     figsize=(4,4), 
                     bins=np.arange(-0.03, 1.03, 0.01), 
                     lw=0.75,
                     alpha=0.2,
                     xlog=False, 
                     ylog=True,
                     density=True, 
                     xlim=(0,1)):

    ref_label = class_labels[reference]
    get_name = {v: k for k, v in class_labels.items()}
    fig, ax = plt.subplots(1, figsize=figsize)    
    for label, score in predictions.items():
        if label == 'datasets': continue
        test = True if label == -1 else False
        sns.histplot(x=score[:, ref_label], 
                     bins=bins, 
                     element="step", 
                     log_scale=(xlog, ylog), 
                     lw = 0 if test else lw, 
                     fill=test, 
                     alpha=alpha if test else 1, 
                     ax=ax, 
                     stat='density' if density else 'count',
                     label=get_name[label]) 
    plt.xlabel(r'score')
    plt.xlim(xlim)
    plt.title(r'Reference class: {}'.format(reference), fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig(workdir+'/classifier_score.png')

# def plot_class_score(test_probs, 
#                      model_probs, 
#                      workdir, 
#                      label=0,
#                      figsize=(4,4), 
#                      bins=np.arange(-0.03, 1.03, 0.01), 
#                      xlog=False, 
#                      ylog=True, 
#                      xlim=(0,1), 
#                      legends=None):

#     fig, ax = plt.subplots(1, figsize=figsize)
#     sns.histplot(x=test_probs[..., label], 
#                  color='k', 
#                  bins=bins, 
#                  element="step", 
#                  log_scale=(xlog, ylog), 
#                  lw=0, 
#                  fill=True, 
#                  alpha=0.2, 
#                  ax=ax, 
#                  label='test')

#     for i, x in enumerate(model_probs):
#         x = x[..., label] 
#         legends = [None] * len(model_probs) if legends is None else legends
#         sns.histplot(x=x, 
#                      bins=bins, 
#                      element="step", 
#                      log_scale=(xlog, ylog), 
#                      lw=0.75, 
#                      fill=False, 
#                      alpha=1, 
#                      ax=ax, 
#                      label=legends[i]) 
#     plt.xlabel(r'score')
#     plt.xlim(xlim)
#     plt.title(r'Classifier, score_label={}'.format(label))
#     plt.legend(loc='upper left')
#     plt.savefig(workdir+'/classifier_score.png')
