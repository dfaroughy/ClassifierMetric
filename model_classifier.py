from src.utils import make_dir, save_configs, save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from src.train import ModelClassifierTest

''' 
Trains a classifier to distinguish between several generative models based on particle-level features.
The classifier is trained on the generated data from each model and evaluated on a reference dataset (JetNet).
'''

###################################################
#...Import model and load configuration cards

from src.architectures import DeepSets as deepsets
from cards.models import DeepSetsConfig as config
classifier_model = deepsets(model_config=config)

# from src.architectures import MLP as mlp
# from cards.models import MLPConfig as config
# classifier_model = mlp(model_config=config)

###################################################

#...Create working folders

directory = '{}.{}.{}feats.{}class.{}batch.{}lr'.format(config.jet_type, config.name, config.dim_input, config.dim_output, config.batch_size, config.lr)
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

#...train and test classifier for discriminating between Models

classifier = ModelClassifierTest(classifier = classifier_model, 
                                datasets = datasets,
                                split_fractions = config.split_fractions,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                workdir = config.workdir,
                                seed = config.seed)

classifier.DataLoaders(batch_size=config.batch_size)
classifier.train()
classifier.test()

#...Evaluate classifier on test datasets

preds={}
label = classifier.predictions[:, -1]
preds['flow-match_mid'] = classifier.predictions[label == 0]
preds['flow-match_eul'] = classifier.predictions[label == 1]
preds['diffusion'] = classifier.predictions[label == 2] 
preds['jetnet'] = classifier.predictions[label == 3]
save_data(preds, workdir=config.workdir, name='predictions')

#...Plot classifier scores

plot_class_score(test_probs=preds['jetnet'], 
                model_probs=[preds['flow-match_mid'], preds['flow-match_eul'], preds['diffusion']],
                label=0,
                workdir=config.workdir+'/results',
                figsize=(5,5), 
                xlim=(1e-5,1),
                bins=50,
                legends=['flow-matching (midpoint)', 'flow-matching (euler)', 'diffusion (DDIM)'])

