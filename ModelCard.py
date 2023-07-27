from ConfigCards import DataConfig, TrainConfig

class MLPConfig(DataConfig, TrainConfig):
    
    name : str = 'MLP'
    dim_input : int = 4
    dim_output : int = 2
    dim_hidden : int = 128 
    num_layers : int = 3 

class DeepSetsConfig(DataConfig, TrainConfig):

    name  = 'DeepSets'
    dim_input : int = 5
    dim_output : int = 2
    dim_hidden : int = 128   
    num_layers_1 : int = 2
    num_layers_2 : int = 2

class ParticleNetConfig(DataConfig, TrainConfig):

    name  = 'ParticleNet'
    dim_input : int = 2
    dim_output : int = 2
    dim_hidden : int = 128 
    num_knn : int = 7     
    dim_conv_1 : int = 32
    dim_conv_2 : int = 64
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    pooling : str = 'average'
    dropout : float = 0.1