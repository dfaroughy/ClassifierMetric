from cards.configs import DataConfig, TrainConfig

class DeepSetsConfig(TrainConfig, DataConfig):

    name = 'DeepSets'
    dim_input = len(DataConfig.features)
    dim_output = len(DataConfig.datasets) - 1
    dim_hidden = 128   
    num_layers_1 = 3
    num_layers_2 = 3

class MLPConfig(TrainConfig, DataConfig):
    
    name = 'MLP'
    dim_input = 5
    dim_output = len(DataConfig.datasets) - 1
    dim_hidden = 128 
    num_layers = 3 

class ParticleNetConfig(TrainConfig, DataConfig):

    name  = 'ParticleNet'
    dim_input = len(DataConfig.features)
    dim_output = len(DataConfig.datasets) - 1
    dim_hidden = 128 
    num_knn = 7     
    dim_conv_1 = 32
    dim_conv_2 = 64
    num_layers_1 = 3
    num_layers_2 = 3
    pooling = 'average'
    dropout = 0.1