import sys

from src.utils import save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from train import ModelClassifierTest
from src.utils import GetConfigs

###################################################
#...Import model and load configuration cards

from models.architectures import DeepSets as deepsets

path = sys.argv[1]
config = GetConfigs(path=path + '/configs.json')
model = deepsets(model_config=config)
ref_class = 'flow_midpoint'
###################################################


datasets = JetNetDataset(dir_path = 'data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.size,
                        preprocess = config.preprocess,
                        particle_features = config.features,
                        )

classifier = ModelClassifierTest(classifier = model, 
                                datasets = datasets,
                                split_fractions = config.split_fractions,
                                epochs = config.epochs, 
                                lr = config.lr, 
                                early_stopping = config.early_stopping,
                                workdir = config.workdir,
                                seed = config.seed)

classifier.dataloader(batch_size=config.batch_size)
classifier.load_model(path=path + '/best_model.pth')
classifier.test(class_labels=config.labels)

plot_class_score(predictions=classifier.predictions,
                class_labels=config.labels,
                reference=ref_class,
                workdir=config.workdir+'/results',
                figsize=(8,8), 
                xlim=(1e-5,1)
                )

print(classifier.log_posterior)