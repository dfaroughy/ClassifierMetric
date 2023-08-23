from ClassifierMetric.utils.plots import plot_class_score
from ClassifierMetric.datamodules.jetnet.datasets import JetNetDataset
from ClassifierMetric.datamodules.jetnet.dataloaders import JetNetDataLoader
from ClassifierMetric.train.trainer import ModelClassifierTest

# configs 

from ClassifierMetric.models.particlenet import ParticleNet
from ClassifierMetric.configs.particlenet_config import ParticleNetConfig as Config

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {
                              'flow_cond' :   ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                              'diff_cond' :   ['diff_tops150_cond_midpoint_100_csts.h5', 'etaphipt_frac'],
                              'flow_uncond' : ['fm_tops150_mp200nfe.h5', 'etaphipt'],
                              'diff_uncond' : ['diff_tops150_midpoint_100_csts.h5', 'etaphipt_frac'],
                              'gan_uncond' :  ['gan_tops150_csts.h5', 'etaphipt_frac'],
                              'jetnet150' :   ['t150.hdf5', 'particle_features']
                              },
                    labels  = {
                              'flow_cond' : 0,
                              'flow_uncond' : 0,
                              'diff_cond' : 1,
                              'diff_uncond' : 1,
                              'gan_uncond' : 2,
                              'jetnet150' : -1    # test data
                              },
                data_split_fracs = [0.6, 0.1, 0.3],
                epochs = 5000,
                batch_size = 2048,
                lr = 0.001,
                early_stopping = 5000,
                warmup_epochs=100,
                dim_hidden = 256, 
                num_knn  = 8,
                dim_conv_1 = 32,
                dim_conv_2 = 64,
                num_layers_1 = 3,
                num_layers_2 = 3,
                dropout = 0.1,
                device = 'cuda:1'
                )

root_dir =  '/home/df630/ClassifierMetric' if 'cuda' in config.device else '/Users/dario/Dropbox/PROJECTS/ML/JetData/ClassifierMetric'

if __name__=="__main__":

    model = ParticleNet(model_config=config)
    config.set_workdir(root_dir + '/results', save_config=True)
    datasets = JetNetDataset(dir_path = root_dir + '/data/', 
                            datasets = config.datasets,
                            class_labels = config.labels,
                            max_num_jets = config.max_num_jets,
                            max_num_constituents = config.max_num_constituents,
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
                     reference='flow_cond',
                     title=config.model_name, 
                     figsize=(8,8), 
                     xlim=(1e-5,1),
                     workdir=config.workdir)