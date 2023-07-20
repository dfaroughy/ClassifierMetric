import torch
import numpy as np
import h5py

from src.utils import make_dir, save_configs, save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataSets
from src.train import MultiClassifierTest

''' 
Trains a classifier to distinguish between two models based on particle-level features.
The two models are:
  1. Flow-Matching
  2. Diffusion
The classifier is trained on the two models and evaluated on a reference dataset (JetNet).

TODO select reference jetnet data in accordance to the train/test split of models
TODO fix ParticleNet 
TODO make a superclass for all Configs
'''

###################################################
#...Import model and configuration cards

from ConfigCard import DataConfig as DataConfig
from ConfigCard import TrainConfig as TrainConfig
from ConfigCard import EvalConfig as EvalConfig
from ModelCard import DeepSetsConfig as ModelConfig
from src.architectures import DeepSets as deepsets

model = deepsets(model_config=ModelConfig)
###################################################

#...Create working folders

directory = '{}.{}.{}.hdims.{}x{}.layers.{}.batch'.format(DataConfig.jet_type,
                                                         ModelConfig.name, 
                                                         ModelConfig.dim_hidden, 
                                                         ModelConfig.num_layers_1, 
                                                         ModelConfig.num_layers_2, 
                                                         TrainConfig.batch_size)

ModelConfig.workdir = make_dir(directory, sub_dirs=['results'], overwrite=False)
save_configs(configs=[DataConfig, TrainConfig, EvalConfig, ModelConfig], filename=ModelConfig.workdir+'/config.json')

Data_train, Data_eval = {}, {}

#...Load data

train_data = JetNetDataSets(dir_path='data/', 
                              data_files=TrainConfig.datafiles,
                              data_class_labels=TrainConfig.labels,
                              num_jets=TrainConfig.size,
                              num_constituents=150, 
                              clip_neg_pt=True,
                              particle_features=['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
                              )

test_data = JetNetDataSets(dir_path='data/', 
                           data_files=EvalConfig.datafiles,
                           num_jets=EvalConfig.size,
                           num_constituents=150, 
                           clip_neg_pt=True,
                           particle_features=['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
                           )

print(train_data.data_summary())
print(test_data.data_summary())

#...Train classifier for discriminating between Model 1 and 2

classifier = MultiClassifierTest(classifier=model, 
                                train_samples=train_data,
                                test_samples=test_data, 
                                epochs=TrainConfig.epochs, 
                                lr=TrainConfig.lr, 
                                early_stopping=TrainConfig.early_stopping,
                                workdir=ModelConfig.workdir,
                                seed=ModelConfig.seed)
classifier.DataLoader(test_size=TrainConfig.test_size, batch_size=TrainConfig.batch_size)
classifier.train()

# #...Evaluate classifier on samples

probs = {}
probs['jetnet'] = classifier.model.probability(Data_eval['jetnet'], batch_size=EvalConfig.batch_size)
probs['flow-match'] = classifier.model.probability(Data_eval['flow-match'], batch_size=EvalConfig.batch_size)
probs['diffusion'] = classifier.model.probability(Data_eval['diffusion'], batch_size=EvalConfig.batch_size)

#...Plot classifier scores

plot_class_score(test_probs=probs['jetnet'], 
                model_probs=[probs['flow-match'], probs['diffusion']],
                label=EvalConfig.class_label,
                workdir=ModelConfig.workdir+'/results',
                figsize=(5,5), 
                xlim=(1e-5,1),
                bins=50,
                legends=['flow-matching', 'diffusion'])

#...Save classifier scores

save_data(probs, workdir=ModelConfig.workdir, name='probs')
