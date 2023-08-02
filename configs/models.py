
# import sys
# sys.path.append('/Users/dario/Dropbox/PROJECTS/ML/JetData/ClassifierMetric')

from dataclasses import dataclass, field
from configs import DataConfig, TrainConfig

@dataclass
class DeepSetsConfig(TrainConfig, DataConfig):

    model_name : str = 'DeepSets'
    dim_input  : int = 5 #field(default_factory = lambda : len(DataConfig.features))
    dim_output : int = 2 #field(default_factory = lambda : len(DataConfig.datasets) - 1)
    dim_hidden : int = 128   
    num_layers_1 : int = 3
    num_layers_2 : int = 3

@dataclass
class ParticleNetConfig(TrainConfig, DataConfig):

    model_name  : str = 'ParticleNet'
    dim_input : int = field(default_factory = lambda : len(DataConfig.features))
    dim_output : int = field(default_factory = lambda : len(DataConfig.datasets) - 1)
    dim_hidden : int = 128 
    num_knn : int = 7     
    dim_conv_1 : int = 32
    dim_conv_2 : int = 64
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    dropout : float = 0.1

@dataclass
class MLPConfig(TrainConfig, DataConfig):

    model_name : str = 'MLP'
    dim_input : int = 5
    dim_output : int = field(default_factory = lambda : len(DataConfig.datasets) - 1)
    dim_hidden : int = 128 
    num_layers : int = 3 


