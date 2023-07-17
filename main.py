import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import json

from src.utils import make_dir, save_configs
from src.plots import plot_class_score
from src.jetnet import JetNetFeatures
from src.train import MultiClassifierTest
from src.architectures import DeepSets

sys.path.append("../")

''' Trains a classifier to distinguish between two models based on particle-level features.
        The two models are:
                1. Flow-Matching
                2. Diffusion
        The classifier is trained on the two models and evaluated on a reference dataset (JetNet).
'''

#...Load config cards

from ConfigCard import DataConfig as Data
from ConfigCard import TrainConfig as Train
from ConfigCard import EvalConfig as Eval

#...Load model config card

from ModelCard import DeepSetsConfig as Config

model = DeepSets(config=Config)

#...Create working folders

directory = '{}.{}.{}.hdims.{}x{}.layers.{}.batch'.format(Data.jet_type,
                                                         Config.name, 
                                                         Config.dim_hidden, 
                                                         Config.num_layers_1, 
                                                         Config.num_layers_2, 
                                                         Train.batch_size)
Config.workdir = make_dir(directory, overwrite=False)
save_configs(configs=[Data, Train, Eval, Config], filename=Config.workdir+'/config.json')

#...Load data

Data_train, Data_eval = {}, {}

#...Data from Model 1: Flow-Matching 

data = np.load(Data.sets['flow-match'])
data = torch.Tensor(data)
data = JetNetFeatures(data)
data.preprocess(methods=Data.preprocess)
Data_train['flow-match'] = data.particles[:Train.size]
Data_eval['flow-match']  = data.particles[Train.size : Train.size + Eval.size]

#...Data for Model 2: Diffusion 

data = h5py.File(Data.sets['diffusion'])
data = torch.tensor(np.array(data['etaphipt_frac'])) 
data = JetNetFeatures(data)
data.preprocess(methods=Data.preprocess)
Data_train['diffusion'] = data.particles[:Train.size] 
Data_eval['diffusion']  = data.particles[Train.size : Train.size + Eval.size]

#...Load and preprocess reference data for evaluation: JetNet 

data = h5py.File(Data.sets['jetnet'])
data = torch.Tensor(np.array(data['particle_features']))
data = JetNetFeatures(data, masked=True)
data.preprocess(methods=Data.preprocess)
Data_eval['jetnet'] = data.particles[:Eval.size]

#...Train classifier for discriminating between Model 1 and 2

classifier = MultiClassifierTest(classifier=model, 
                                samples=(Data_train['flow-match'], Data_train['diffusion']), 
                                epochs=Train.epochs, 
                                lr=Train.lr, 
                                early_stopping=Train.early_stopping,
                                workdir=Config.workdir,
                                seed=Config.seed)
classifier.DataLoader(test_size=Train.test_size, batch_size=Train.batch_size)
classifier.train()

#...Evaluate classifier on samples

Data.probs = {}
Data.probs['jetnet'] = classifier.model.probability(Data_eval['jetnet'], batch_size=Eval.batch_size)
Data.probs['flow-match'] = classifier.model.probability(Data_eval['flow-match'], batch_size=Eval.batch_size)
Data.probs['diffusion'] = classifier.model.probability(Data_eval['diffusion'], batch_size=Eval.batch_size)

print(Data.probs)

#...Plot classifier scores

plot_class_score(test=Data.probs['jetnet'], 
                models=[Data.probs['flow-match'], 
                        Data.probs['diffusion']],
                label=Eval.class_label,
                workdir=Config.workdir,
                figsize=(5,5), 
                xlim=(1e-5,1),
                bins=50,
                legends=['flow-matching', 'diffusion'])

# store results: