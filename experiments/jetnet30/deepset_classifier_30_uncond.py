from ClassifierMetric.utils.plots import plot_class_score
from ClassifierMetric.datamodules.jetnet.datasets import JetNetDataset
from ClassifierMetric.datamodules.jetnet.dataloaders import JetNetDataLoader
from ClassifierMetric.utils.trainer import ModelClassifierTest

from ClassifierMetric.models.deepsets import DeepSets
from ClassifierMetric.configs.deepsets_config import DeepSetsConfig as Config

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {
                              'flow_midpoint' : ['fm_tops30_mp200nfe.h5', 'etaphipt'],
                              'diff_midpoint' : ['diff_tops30_midpoint_100_csts.h5', 'etaphipt_frac'],
                              'flow_euler' :    ['fm_tops30_eu200nfe.h5', 'etaphipt'],
                              'diff_euler' :    ['diff_tops30_euler_200_csts.h5', 'etaphipt_frac'],
                              'diff_em' :       ['diff_tops30_em_200_csts.h5', 'etaphipt_frac'] ,
                              'jetnet30' :      ['t.hdf5', 'particle_features']
                              },
                    labels  = {
                              'flow_midpoint' : 0, 
                              'diff_midpoint' : 1,
                              'flow_euler' : 2,
                              'diff_euler' : 3,
                              'diff_em' : 4,
                              'jetnet30' : -1 # test data
                              },
                data_split_fracs = [0.6, 0.1, 0.3],
                epochs = 1000,
                num_constituents=30,
                batch_size = 2048,
                warmup_epochs= 50,
                dim_hidden = 128, 
                num_layers_1 = 3,
                num_layers_2 = 3,
                device = 'cuda:1'
                )

if __name__=="__main__":

    deepset = DeepSets(model_config=config)
    config.save(path=config.workdir + '/configs.json')
    datasets = JetNetDataset(dir_path = '/home/df630/ClassifierMetric/data/', 
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
    classifier.train()
    classifier.test(class_labels=config.labels)

    plot_class_score(predictions=classifier.predictions,
                     class_labels=config.labels,
                     reference='flow_midpoint',
                     title=config.model_name, 
                     figsize=(8,8), 
                     xlim=(1e-5,1),
                     workdir=config.workdir,
                     legend_loc='upper right')