# Model config card for the Classifier

class BaseConfig:
    device : str = 'cuda:0'
    seed : int = 12345


class MLPConfig(BaseConfig):
    
    name : str = 'MLP'
    dim_input : int = 4
    dim_output : int = 2
    dim_hidden : int = 128 
    num_layers : int = 3 

class DeepSetsConfig(BaseConfig):

    name  = 'DeepSets'
    dim_input : int = 5
    dim_output : int = 2
    dim_hidden : int = 128   
    num_layers_1 : int = 3
    num_layers_2 : int = 3 

class ParticleNetConfig(BaseConfig):

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