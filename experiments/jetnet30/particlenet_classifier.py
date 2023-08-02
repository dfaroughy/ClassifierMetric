from ClassifierMetric.utils.plots import plot_class_score
from ClassifierMetric.datamodules.jetnet.datasets import JetNetDataset
from ClassifierMetric.datamodules.jetnet.dataloaders import JetNetDataLoader
from ClassifierMetric.utils.trainer import ModelClassifierTest

from ClassifierMetric.models.particlenet import ParticleNet
from ClassifierMetric.configs.particlenet_config import ParticleNetConfig as Config

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

particlenet = ParticleNet(model_config=config)
config.save(path=config.workdir + '/configs.json')
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
classifier = ModelClassifierTest(classifier = particlenet, 
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