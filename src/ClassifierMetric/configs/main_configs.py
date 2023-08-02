
from dataclasses import dataclass, field
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
    size : int = None # if None, use all data
    batch_size : int = 1024
    epochs : int = 10000   
    early_stopping : int = 30 
    warmup_epochs : int = 100    
    lr : float = 0.001
    seed : int = 12345

