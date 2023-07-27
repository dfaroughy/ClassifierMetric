import torch
import numpy as np
import h5py

from src.utils import make_dir, save_configs, save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from src.train import MultiClassifierTest

''' 
Trains a classifier to distinguish between two models based on particle-level features.
The two models are:
  1. Flow-Matching
  2. Diffusion
The classifier is trained on the two models and evaluated on a reference dataset (JetNet).
TODO fix ParticleNet 
'''

###################################################
#...Import model and configuration cards

from cards.ModelCard import DeepSetsConfig as config
from src.architectures import DeepSets as deepsets

classifier_model = deepsets(model_config=config)

###################################################

#...Create working folders

directory = '{}.{}.{}.hdims.{}x{}.layers.{}.batch'.format(config.jet_type,
                                                         config.name, 
                                                         config.dim_hidden, 
                                                         config.num_layers_1, 
                                                         config.num_layers_2, 
                                                         config.batch_size)

config.workdir = make_dir(directory, sub_dirs=['results'], overwrite=False)
save_configs(configs=[config], filename=config.workdir+'/configs.json')

#...Load data

datasets = JetNetDataset(dir_path = 'data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.size,
                        preprocess = config.preprocess,
                        particle_features = config.features,
                        )

#...Train classifier for discriminating between Modelss

classifier = MultiClassifierTest(classifier = classifier_model, 
                                samples = datasets,
                                split_fractions = config.split_fractions,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                workdir = config.workdir,
                                seed = config.seed)

classifier.DataLoader(batch_size=config.batch_size)
classifier.train()

# #...Evaluate classifier on samples

# probs = {}
# probs['jetnet'] = classifier.config.probability(Data_eval['jetnet'], batch_size=EvalConfig.batch_size)
# probs['flow-match'] = classifier.config.probability(Data_eval['flow-match'], batch_size=EvalConfig.batch_size)
# probs['diffusion'] = classifier.config.probability(Data_eval['diffusion'], batch_size=EvalConfig.batch_size)

# #...Plot classifier scores

# plot_class_score(test_probs=probs['jetnet'], 
#                 model_probs=[probs['flow-match'], probs['diffusion']],
#                 label=EvalConfig.class_label,
#                 workdir=config.workdir+'/results',
#                 figsize=(5,5), 
#                 xlim=(1e-5,1),
#                 bins=50,
#                 legends=['flow-matching', 'diffusion'])

# #...Save classifier scores

# save_data(probs, workdir=config.workdir, name='probs')
