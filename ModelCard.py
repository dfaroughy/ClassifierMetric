# Model config card for the Classifier

class BaseConfig:
    device : str = 'cpu'
    seed : int = 12345


class MLPConfig(BaseConfig):
    
    name : str = 'MLP'
    dim_input : int = 4
    dim_output : int = 2
    dim_hidden : int = 128 
    num_layers : int = 3 

class DeepSetsConfig(BaseConfig):

    name  = 'DeepSets'
    dim_input : int = 4
    dim_output : int = 2
    dim_hidden : int = 128   
    num_layers_1 : int = 3 
    num_layers_2 : int = 3 

class ParticleNetConfig(BaseConfig):

    name  = 'ParticleNet'
    dim_input : int = 2
    dim_output : int = 2
    dim_hidden : int = 128 
    num_knn : int = 16     
    dim_conv_1 : int = 64
    dim_conv_2 : int = 64
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    dropout : float = 0.1