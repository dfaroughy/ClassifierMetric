from src.utils import make_dir
from src.plots import plot_class_score
from src.datamodule.datasets import JetNetDataset
from src.datamodule.dataloaders import JetNetDataLoader
from src.trainer import ModelClassifierTest


''' 
Trains a classifier to distinguish between several generative models based on particle-level features.
The classifier is trained on the generated data from each model and evaluated on a reference dataset (JetNet).

TODO fix preprocess = None case
TODO fix paths for src, data, results

'''

###################################################

from config.configs import DeepSetsConfig as Config
from src.models.deepsets import DeepSets

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {'flow_midpoint' : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                                'diff_midpoint' : ['midpoint_100_csts.h5', 'etaphipt_frac']},
                labels      = {'flow_midpoint' : 0, 'diff_midpoint' : 1},
                data_split_fracs = [0.5, 0.2, 0.3],
                size = 100000,
                epochs = 5,
                dim_hidden  = 128,   
                num_layers_1 = 2,
                num_layers_2 = 3,
                device = 'cpu'
                )

model = DeepSets(model_config=config)

###################################################


#...create working dir in results/

directory = '{}.{}.{}feats.{}class.{}batch'.format(config.data_name, config.model_name, config.dim_input, config.dim_output, config.batch_size)
config.workdir = make_dir('results/' + directory, overwrite=True)
config.save(path=config.workdir+'/configs.json')

#...definte datasets, dataloader and classifier model

datasets = JetNetDataset(dir_path = 'data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.size,
                        preprocess = config.preprocess,
                        particle_features = config.features,
                        compute_jet_features=False,
                        remove_negative_pt = True
                        ) 
dataloader = JetNetDataLoader(datasets=datasets, data_split_fracs=config.data_split_fracs, batch_size=config.batch_size)
classifier = ModelClassifierTest(classifier = model, 
                                dataloader = dataloader,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                warmup_epochs = config.warmup_epochs,
                                workdir = config.workdir,
                                seed = config.seed)

if __name__=="__main__":

    classifier.train()
    classifier.test(class_labels=config.labels)

    plot_class_score(predictions=classifier.predictions,
                    class_labels=config.labels,
                    reference='flow_midpoint',
                    figsize=(8,8), 
                    xlim=(1e-5,1),
                    workdir=config.workdir)