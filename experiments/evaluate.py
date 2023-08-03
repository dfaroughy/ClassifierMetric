import sys
from ClassifierMetric.utils.plots import plot_class_score
from ClassifierMetric.datamodules.jetnet.datasets import JetNetDataset
from ClassifierMetric.datamodules.jetnet.dataloaders import JetNetDataLoader
from ClassifierMetric.utils.trainer import ModelClassifierTest

from ClassifierMetric.models.particlenet import ParticleNet
from ClassifierMetric.configs.particlenet_config import ParticleNetConfig

workdir = sys.argv[1]
config = ParticleNetConfig.load(path=workdir + '/configs.json')
config.workdir = workdir

model = ParticleNet(model_config=config)
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
classifier = ModelClassifierTest(classifier = model, 
                                dataloader = dataloader,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                warmup_epochs = config.warmup_epochs,
                                workdir = config.workdir,
                                seed = config.seed)

if __name__=="__main__":

    classifier.load_model(path=config.workdir + '/best_model.pth')
    classifier.test(class_labels=config.labels)     

    plot_class_score(predictions=classifier.predictions,
                    class_labels=config.labels,
                    reference='flow_midpoint',
                    figsize=(8,8), 
                    xlim=(1e-5,1),
                    workdir=config.workdir)
    
    print(classifier.predictions)
