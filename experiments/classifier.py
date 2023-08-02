from src.utils import make_dir, save_configs, save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from train import ModelClassifierTest

''' 
Trains a classifier to distinguish between several generative models based on particle-level features.
The classifier is trained on the generated data from each model and evaluated on a reference dataset (JetNet).

TODO fix preprocess = None case

'''

###################################################
#...Import model and load configuration cards

# from src.architectures import DeepSets as deepsets
# from cards.models import DeepSetsConfig as config
# model = deepsets(model_config=config)

from models.architectures import ParticleNet as particlenet
from configs.models import ParticleNetConfig as config
model = particlenet(model_config=config)

###################################################

#...Create working folders

directory = '{}.{}.{}feats.{}class.{}batch.{}lr'.format(config.jet_type, config.name, config.dim_input, config.dim_output, config.batch_size, config.lr)
config.workdir = make_dir(directory, sub_dirs=['results'], overwrite=True)
save_configs(configs=[config], filename=config.workdir+'/configs.json')

#...Load data and train model classifier 

datasets = JetNetDataset(dir_path = 'data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.size,
                        preprocess = config.preprocess,
                        particle_features = config.features,
                        compute_jet_features=False,
                        remove_negative_pt = True
                        ) 

classifier = ModelClassifierTest(classifier = model, 
                                datasets = datasets,
                                split_fractions = config.split_fractions,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                warmup_epochs = config.warmup_epochs,
                                workdir = config.workdir,
                                seed = config.seed)

classifier.dataloader(batch_size=config.batch_size)
classifier.train()

#...Evaluate classifier on test datasets

classifier.test(class_labels=config.labels)
plot_class_score(predictions=classifier.predictions,
                class_labels=config.labels,
                reference='flow_midpoint',
                workdir=config.workdir+'/results',
                figsize=(8,8), 
                xlim=(1e-5,1)
                )