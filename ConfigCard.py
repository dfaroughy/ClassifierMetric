class DataConfig:

    jet_type : str = 'tops'

    sets : dict = {
                   'jetnet'     : 'data/t150.hdf5',
                   'flow-match' : 'data/generated_data_FM.npy',
                   'diffusion'  : 'data/ddim_200.h5'
                   }

    preprocess: dict = {
                        # 'log_pt': {},
                        'normalize': {},
                        'logit_transform': {'alpha' : 1e-5},
                        'standardize': {'sigma' : 1.0}
                        }
  
class TrainConfig:

    size : int = 150000
    test_size : float = 0.7    
    batch_size : int = 1024
    epochs : int = 3   
    early_stopping : int = 30     
    lr : float = 0.0005

class EvalConfig:

    size : int = 50000  
    batch_size : int = 1024 
    class_label : int = 0
