from ClassifierMetric.utils.plots import plot_class_score
from ClassifierMetric.datamodules.jetnet.datasets import JetNetDataset
from ClassifierMetric.datamodules.jetnet.dataloaders import JetNetDataLoader
from ClassifierMetric.utils.trainer import ModelClassifierTest

# configs 

from ClassifierMetric.models.particlenet import ParticleNet
from ClassifierMetric.configs.particlenet_config import ParticleNetConfig as Config

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {
                              'flow_midpoint' : ['fm_tops150_mp200nfe.h5', 'etaphipt'],
                              'diff_midpoint' : ['diff_tops150_midpoint_100_csts.h5', 'etaphipt_frac'],
                              'flow_euler' :    ['fm_tops150_eu200nfe.h5', 'etaphipt'],
                              'diff_euler' :    ['diff_tops150_euler_200_csts.h5', 'etaphipt_frac'],
                              'diff_em' :       ['diff_tops150_em_200_csts.h5', 'etaphipt_frac'],  
                              'diff_ddim' :     ['diff_tops150_ddim_200_csts.h5', 'etaphipt_frac'],
                              'jetnet150' :     ['t150.hdf5', 'particle_features'],
                              },
                    labels  = {
                              'flow_midpoint' : 0, 
                              'diff_midpoint' : 1,
                              'flow_euler' : 2,
                              'diff_euler' : 3,
                              'diff_em' : 4,
                              'diff_ddim' : 5,
                              'jetnet150' : -1 # test data
                              },
                data_split_fracs = [0.6, 0.1, 0.3],
                epochs = 1000,
                batch_size = 2048,
                warmup_epochs= 50,
                dim_hidden = 256, 
                num_knn  = 7,
                dim_conv_1 = 32,
                dim_conv_2 = 64,
                num_layers_1 = 3,
                num_layers_2 = 3,
                device = 'cuda:0'
                )

root_dir =  '/home/df630/ClassifierMetric' if 'cuda' in config.device else '/Users/dario/Dropbox/PROJECTS/ML/JetData/ClassifierMetric'

if __name__=="__main__":

    model = ParticleNet(model_config=config)
    config.set_workdir(root_dir + '/results', save_config=True)
    datasets = JetNetDataset(dir_path = root_dir + '/data/', 
                            datasets = config.datasets,
                            class_labels = config.labels,
                            num_jets = config.max_num_jets,
                            num_constituents = config.max_num_constituents,
                            preprocess = config.preprocess,
                            particle_features = config.features,
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
    classifier.train()
    classifier.test(class_labels=config.labels)

    plot_class_score(predictions=classifier.predictions,
                     class_labels=config.labels,
                     reference='flow_midpoint',
                     title=config.model_name, 
                     figsize=(8,8), 
                     xlim=(1e-5,1),
                     workdir=config.workdir)