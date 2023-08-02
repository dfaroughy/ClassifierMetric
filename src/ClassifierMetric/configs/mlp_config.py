import json
from dataclasses import dataclass, asdict
from ClassifierMetric.utils import make_dir
from ClassifierMetric.configs.main_configs import TrainConfig, DataConfig

@dataclass
class MLPConfig(TrainConfig, DataConfig):

    model_name : str = 'MLP'
    dim_input : int = 5
    dim_output : int = 2
    dim_hidden : int = 128 
    num_layers : int = 3 
    mkdir : bool = True

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features)
        self.dim_output = len(self.datasets) - 1
        if self.mkdir:
            self.workdir = make_dir('results/{}.{}'.format(self.data_name, self.model_name), overwrite=False)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: data = json.load(json_file)
        return cls(**data)