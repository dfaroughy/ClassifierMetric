
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict

@dataclass
class DataConfig:

    data_name : str = 'tops'
    features   : List[str] = field(default_factory = lambda : ['eta_rel', 'phi_rel'])
    preprocess : List[str] = field(default_factory = lambda : ['standardize'])
    datasets   : Dict[str, List[str]] = field(default_factory = lambda: {'jetnet150': ['t150.hdf5', 'particle_features']})
    labels : Dict[str, int] = field(default_factory = lambda:  {'jetnet150': -1} )
    
    def __post_init__(self):
        coords = ['eta_rel', 'phi_rel']
        self.features = [feature for feature in coords if feature not in self.features] + self.features
        testset = {'jetnet150': ['t150.hdf5', 'particle_features']}
        testlabel = {'jetnet150': -1}
        testset.update(self.datasets)
        testlabel.update(self.labels)
        self.datasets = testset
        self.labels = testlabel
    
@dataclass
class TrainConfig:

    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [0.5, 0.2, 0.3])  # train / val / test 
    size : int = None 
    batch_size : int = 1024
    epochs : int = 10000   
    early_stopping : int = 30 
    warmup_epochs : int = 100    
    lr : float = 0.001
    seed : int = 12345


@dataclass
class DeepSetsConfig(TrainConfig, DataConfig):

    #...model parameters

    model_name : str = 'DeepSets'
    dim_input  : int = 2 
    dim_output : int = 2
    dim_hidden : int = 128   
    num_layers_1 : int = 3
    num_layers_2 : int = 3

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features)
        self.dim_output = len(self.datasets) - 1

    def save(self, path):
        with open(path, 'w') as f: json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: data = json.load(json_file)
        return cls(**data)

@dataclass
class ParticleNetConfig(TrainConfig, DataConfig):

    #...model parameters

    model_name  : str = 'ParticleNet'
    dim_input : int = 2
    dim_output : int = 2
    dim_hidden : int = 128 
    num_knn : int = 7     
    dim_conv_1 : int = 32
    dim_conv_2 : int = 64
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    dropout : float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features)
        self.dim_output = len(self.datasets) - 1

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: data = json.load(json_file)
        return cls(**data)
    
@dataclass
class MLPConfig(TrainConfig, DataConfig):

    #...model parameters

    model_name : str = 'MLP'
    dim_input : int = 5
    dim_output : int = 2
    dim_hidden : int = 128 
    num_layers : int = 3 

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features)
        self.dim_output = len(self.datasets) - 1

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: data = json.load(json_file)
        return cls(**data)