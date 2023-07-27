class DataConfig:

    jet_type = 'tops'

    datasets = {'jetnet'      : ('t150.hdf5', 'particle_features'),
                'fm_midpoint' : ('fm_tops150_mp200nfe.h5', 'etaphipt'),
                'diff_ddim'   : ('ddim_200.h5', 'etaphipt_frac')}
    
    labels = {'fm_midpoint' : 0,
              'diff_ddim'   : 1,
              'jetnet'      : 2} 
    
    features = ['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
    preprocess = ['center_jets', 'standardize']
    
class TrainConfig:

    device = 'cpu'
    split_fractions = [0.4, 0.3, 0.3]  # train / val / test 
    size = 10000 
    batch_size  = 200
    epochs  = 10   
    early_stopping  = 30     
    lr  = 0.0005
    seed = 12345
