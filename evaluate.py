import sys

from src.utils import save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from src.train import ModelClassifierTest
from src.utils import GetConfigs

###################################################
#...Import model and load configuration cards

from src.architectures import DeepSets as deepsets

path = sys.argv[1]
config = GetConfigs(path=path + '/configs.json')
model = deepsets(model_config=config)

###################################################

#...Load data and model 

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

classifier.DataLoaders(batch_size=config.batch_size)
classifier.load_model(path=path + '/best_model.pth')

#...Evaluate classifier on test datasets

classifier.test()

preds={}
label = classifier.predictions[:, -1]
preds['jetnet'] = classifier.predictions[label == -1]
preds['flow (midpoint)'] = classifier.predictions[label == 0][:preds['jetnet'].shape[0]] 
preds['diffusion (midpoint)'] = classifier.predictions[label == 1][:preds['jetnet'].shape[0]] 
preds['flow (euler)'] = classifier.predictions[label == 2][:preds['jetnet'].shape[0]] 
preds['diffusion (euler)'] = classifier.predictions[label == 3][:preds['jetnet'].shape[0]]
save_data(preds, workdir=config.workdir, name='predictions')

#...Plot classifier scores

plot_class_score(test_probs=preds['jetnet'], 
                model_probs=[preds['flow (midpoint)'], preds['diffusion (midpoint)'], preds['flow (euler)'], preds['diffusion (euler)']],
                label=0,
                workdir=config.workdir+'/results',
                figsize=(8,8), 
                xlim=(1e-5,1),
                legends=['flow-matching (midpoint)', 'diffusion (midpoint)', 'flow-matching (euler)', 'diffusion (euler)']
                )