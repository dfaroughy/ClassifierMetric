class DataConfig:

    jet_type = 'tops'

    datasets = {'jetnet'      : ('t150.hdf5', 'particle_features'),
                'fm_midpoint' : ('fm_tops150_mp200nfe.h5', 'etaphipt'),
                'fm_euler'    : ('fm_tops150_eu200nfe.h5', 'etaphipt'),
                'diff_ddim'   : ('ddim_200.h5', 'etaphipt_frac')}
    
    labels = {'fm_midpoint' : 0,
              'fm_euler'    : 1,
              'diff_ddim'   : 2,
              'jetnet'      : 3} 
    
    features = ['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
    preprocess = ['center_jets', 'standardize']
    
class TrainConfig:

    device = 'cpu'
    split_fractions = [0.4, 0.3, 0.3]  # train / val / test 
    size = 200000 
    batch_size  = 1000
    epochs = 5   
    early_stopping  = 30     
    lr  = 0.0005
    seed = 12345
