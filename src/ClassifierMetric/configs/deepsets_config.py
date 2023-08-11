import json
from datetime import datetime
from dataclasses import dataclass, asdict
from ClassifierMetric.utils.utils import make_dir, print_table
from ClassifierMetric.configs.base_configs import TrainConfig, DataConfig

@dataclass
class DeepSetsConfig(TrainConfig, DataConfig):

    ###############################
    model_name : str = 'DeepSets'
    dim_input  : int = 2 
    dim_output : int = 2
    dim_hidden : int = 256   
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    ###############################

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features)
        self.dim_output = len(self.datasets) - 1

    def set_workdir(self, path: str='.', dir_name: str=None, save_config: bool=True):
        time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
        dir_name = '{}.{}.{}_{}'.format(self.model_name, self.data_name, self.max_num_constituents, time) if dir_name is None else dir_name
        self.workdir = make_dir(path + '/' + dir_name, overwrite=False)
        if save_config: self.save()

    def save(self, path: str=None):
        config = asdict(self)
        print_table(config)
        path = self.workdir + '/config.json' if path is None else path
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: config = json.load(json_file)
        print_table(config)
        # config['mkdir'] = False
        return cls(**config)