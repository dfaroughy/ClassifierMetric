# import sys
# sys.path.append('/Users/dario/Dropbox/PROJECTS/ML/JetData/ClassifierMetric')

from src.plots import plot_class_score
from src.datamodule.datasets import JetNetDataset
from src.datamodule.dataloaders import JetNetDataLoader
from src.trainer import ModelClassifierTest
from src.models.deepsets import DeepSets
from config.configs import DeepSetsConfig as Config

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {'flow_midpoint' : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                                'diff_midpoint' : ['midpoint_100_csts.h5', 'etaphipt_frac']},
                labels      = {'flow_midpoint' : 0, 'diff_midpoint' : 1},
                data_split_fracs = [0.5, 0.2, 0.3],
                size = 10000,
                epochs = 10,
                device = 'cpu',
                mkdir = True
                )

deepset = DeepSets(model_config=config)
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
dataloader = JetNetDataLoader(datasets=datasets, data_split_fracs=config.data_split_fracs, batch_size=config.batch_size)
classifier = ModelClassifierTest(classifier = deepset, 
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