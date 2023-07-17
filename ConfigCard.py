class DataConfig:

    jet_type : str = 'tops'

    sets : dict = {
                   'jetnet'     : 'data/t150.hdf5',
                   'flow-match' : 'data/generated_data_FM.npy',
                   'diffusion'  : 'data/ddim_200.h5'
                   }

    preprocess: dict = {
                        # 'normalize': {},
                        'log_pt':{},
                        # 'logit_transform': {'alpha' : 1e-5},
                        'standardize': {'sigma' : 5.0}
                        }
  
class TrainConfig:

    size : int = 15000
    test_size : float = 0.7    
    batch_size : int = 5000
    epochs : int = 3   
    early_stopping : int = 30     
    lr : float = 0.0005

class EvalConfig:

    size : int = 5000  
    batch_size : int = 100 
    class_label : int = 0
