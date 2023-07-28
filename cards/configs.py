class DataConfig:

    jet_type = 'tops'

    datasets = {'truth'     : ['t150.hdf5', 'particle_features'],
                'flow_mid'  : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                'diff_mid'  : ['diff_midpoint_100_csts.h5', 'etaphipt_frac']}
    
    labels = {'flow_mid'   : 0,
              'diff_mid'   : 1,
              'truth'      : 2} 
    
    features = ['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
    preprocess = ['standardize']
    
class TrainConfig:

    device = 'cpu'
    split_fractions = [0.4, 0.3, 0.3]  # train / val / test 
    size = 220000 
    batch_size  = 1024
    epochs = 10000   
    early_stopping  = 20     
    lr  = 0.0001
    seed = 12345
