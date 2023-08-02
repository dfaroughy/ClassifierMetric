import sys
sys.path.append('/Users/dario/Dropbox/PROJECTS/ML/JetData/ClassifierMetric')

from src.utils import make_dir, save_configs
from src.plots import plot_class_score
from src.datamodule.datasets import JetNetDataset
from src.datamodule.dataloaders import JetNetDataLoader
from src.trainer import ModelClassifierTest


''' 
Trains a classifier to distinguish between several generative models based on particle-level features.
The classifier is trained on the generated data from each model and evaluated on a reference dataset (JetNet).

TODO fix preprocess = None case

'''

###################################################
#...Import model and load configuration cards

from src.models.deepsets import DeepSets
from configs.configs import DeepSetsConfig

config = DeepSetsConfig(features   = ['pt_rel', 'R'],
                        preprocess = ['standardize'],
                        datasets   = {'flow_midpoint'  : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                                      'diff_midpoint'  : ['midpoint_100_csts.h5', 'etaphipt_frac']},
                        labels = {'flow_midpoint' : 0, 'diff_midpoint' : 1},
                        size = 10000
                        )

model = DeepSets(model_config=config)
###################################################

directory = '{}.{}.{}feats.{}class.{}batch'.format(config.data_name, config.model_name, config.dim_input, config.dim_output, config.batch_size)
config.workdir = make_dir('results/' + directory, sub_dirs=['results'], overwrite=True)

if __name__=="__main__":

    config.save(path=config.workdir+'/configs.json')

    datasets = JetNetDataset(dir_path = 'data/', 
                            datasets = config.datasets,
                            class_labels = config.labels,
                            num_jets = config.size,
                            preprocess = config.preprocess,
                            particle_features = config.features,
                            compute_jet_features=False,
                            remove_negative_pt = True
                            ) 

    dataloader = JetNetDataLoader(datasets=datasets,
                                  data_split_fracs=config.split_fractions,
                                  batch_size=config.batch_size)

    classifier = ModelClassifierTest(classifier = model, 
                                    dataloader = dataloader,
                                    # split_fractions = config.split_fractions,
                                    epochs = config.epochs, 
                                    lr = config.lr, 
                                    early_stopping = config.early_stopping,
                                    warmup_epochs = config.warmup_epochs,
                                    workdir = config.workdir,
                                    seed = config.seed)

    # classifier.dataloader(batch_size=config.batch_size)
    

    #...train and evaluate classifier on test datasets

    classifier.train()
    classifier.test(class_labels=config.labels)

    plot_class_score(predictions=classifier.predictions,
                    class_labels=config.labels,
                    reference='flow_midpoint',
                    workdir=config.workdir+'/results',
                    figsize=(8,8), 
                    xlim=(1e-5,1)
                    )