class DataConfig:

    jet_type = 'tops'

    datasets = {
                'flow_midpoint'  : ['fm_tops150_cond_mp200nfe.h5', 'etaphipt'],
                'diff_midpoint'  : ['midpoint_100_csts.h5', 'etaphipt_frac'],
                'flow_euler'     : ['fm_tops150_cond_eu200nfe.h5', 'etaphipt'],
                'diff_euler'     : ['euler_200_csts.h5', 'etaphipt_frac'],
                'jetnet150'      : ['t150.hdf5', 'particle_features']
                }
    labels = {
              'flow_midpoint'   : 0,
              'diff_midpoint'   : 1,
              'flow_euler'      : 2,
              'diff_euler'      : 3,
              'jetnet150'       : -1  # test dataset
              } 
    features = ['eta_rel', 'phi_rel', 'pt_rel', 'R', 'e_rel']
    preprocess = ['normalize', 'logit_transform', 'standardize']
    
class TrainConfig:

    device = 'cuda:1'
    split_fractions = [0.5, 0.2, 0.3]  # train / val / test 
    size = None 
    batch_size  = 4096
    epochs = 10000   
    early_stopping  = 30 
    warmup_epochs = 100    
    lr  = 0.001
    seed = 12345
