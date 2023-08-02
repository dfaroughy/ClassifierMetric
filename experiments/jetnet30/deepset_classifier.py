from ClassifierMetric.plots import plot_class_score
from ClassifierMetric.datamodule.datasets import JetNetDataset
from ClassifierMetric.datamodule.dataloaders import JetNetDataLoader
from ClassifierMetric.trainer import ModelClassifierTest

from ClassifierMetric.models.deepsets import DeepSets
from ClassifierMetric.configs.deepsets_config import DeepSetsConfig as Config

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {'flow_midpoint' : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                               'diff_midpoint' : ['midpoint_100_csts.h5', 'etaphipt_frac'],
                               'flow_euler' : ['fm_tops150_cond_eu200nfe.h5', 'etaphipt'],
                               'diff_euler' : ['euler_200_csts.h5', 'etaphipt_frac']},
                labels      = {'flow_midpoint' : 0, 
                               'diff_midpoint' : 1,
                               'flow_euler' : 2,
                               'diff_euler' : 3},
                data_split_fracs = [0.5, 0.2, 0.3],
                epochs = 1000,
                num_constituents = 30,
                device = 'cpu'
                )

deepset = DeepSets(model_config=config)
config.save(path=config.workdir+'/configs.json')
datasets = JetNetDataset(dir_path = '../../data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.num_jets,
                        num_constituents = config.num_constituents,
                        preprocess = config.preprocess,
                        particle_features = config.features,
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