from src.utils import save_data
from src.plots import plot_class_score
from src.datasets import JetNetDataset
from src.train import ModelClassifierTest
from src.utils import GetConfigs

###################################################
#...Import model and load configuration cards

from src.architectures import DeepSets as deepsets

path = 'tops.DeepSets.5feats.2class.1024batch.0.0001lr__1'
config = GetConfigs(path=path + '/configs.json')
classifier_model = deepsets(model_config=config)

###################################################

#...Load data and model 

datasets = JetNetDataset(dir_path = 'data/', 
                        datasets = config.datasets,
                        class_labels = config.labels,
                        num_jets = config.size,
                        preprocess = config.preprocess,
                        particle_features = config.features,
                        )

classifier = ModelClassifierTest(classifier = classifier_model, 
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
preds['jetnet'] = classifier.predictions[label == 2]
preds['flow-match_mid'] = classifier.predictions[label == 0][:preds['jetnet'].shape[0]] 
preds['diffusion'] = classifier.predictions[label == 1][:preds['jetnet'].shape[0]] 
save_data(preds, workdir=config.workdir, name='predictions')

#...Plot classifier scores

plot_class_score(test_probs=preds['jetnet'], 
                model_probs=[preds['flow-match_mid'], preds['diffusion']],
                label=0,
                workdir=config.workdir+'/results',
                figsize=(5,5), 
                xlim=(1e-5,1),
                legends=['flow-matching', 'diffusion'])

#...perofmance profiling

# import cProfile, pstats, io
# with cProfile.Profile() as pr:
#     classifier.DataLoaders(batch_size=config.batch_size)
#     classifier.load_model(path=path + '/best_model.pth')
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
# ps.print_stats()
# print(s.getvalue())