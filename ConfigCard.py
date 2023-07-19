class DataConfig:

    jet_type : str = 'tops'
    preprocess: dict = {'normalize': {},
                        'logit_transform': {'alpha' : 1e-5},
                        'standardize': {'sigma' : 1.0}}
class TrainConfig:

    datafiles : dict = {'fm_midpoint' : ('fm_tops150_mp200nfe.h5', 'etaphipt'),
                        'fm_euler'    : ('fm_tops150_eu200nfe.h5', 'etaphipt'),
                        'diff_ddim'   : ('ddim_200.h5', 'etaphipt_frac')}
                        
    labels : dict = {'fm_midpoint' : 0,
                        'fm_euler' : 1,
                        'diff_ddim': 2}
    size : int = None
    test_size : float = 0.7    
    batch_size : int = 1024
    epochs : int = 10000   
    early_stopping : int = 30     
    lr : float = 0.0005

class EvalConfig:

    datafiles : dict = {'jetnet' : ('t150.hdf5', 'particle_features')}
    size : int = None  
    test_size : float = 0.7
    batch_size : int = 1024 
    class_label : int = 0
