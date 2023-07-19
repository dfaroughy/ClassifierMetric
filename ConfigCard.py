class DataConfig:

    jet_type : str = 'tops'

    sets : dict = {
                   'flow-match' : ('generated_data_FM.npy', None),
                   'diffusion'  : ('ddim_200.h5', 'etaphipt_frac'),
                   'jetnet'     : ('t150.hdf5', 'particle_features')
                   }

    preprocess: dict = {
                        'normalize': {},
                        'logit_transform': {'alpha' : 1e-5},
                        'standardize': {'sigma' : 1.0}
                        }
  
class TrainConfig:

    size : int = 150000
    test_size : float = 0.7    
    batch_size : int = 1024
    epochs : int = 10000   
    early_stopping : int = 30     
    lr : float = 0.0005

class EvalConfig:

    size : int = 50000  
    batch_size : int = 1024 
    class_label : int = 0
