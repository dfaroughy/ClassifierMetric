from torch.utils.data import Dataset
import torch
import numpy as np
import os
import h5py

class JetNetDataLoad(Dataset):

    ''' Loads data from data files in format `.npy` or `.hdf5`.
        Adds zero padding to each event up to 'num_constituents'.
        Adds mask to each event to indicate which particles are padded.
        Orders particles in each event by descending pt.

        - args: `dir_path`, `num_constituents`
        - output: torch.tensor with shape `(N, num_constituents, 4)` 
        - features: [`eta_rel`, `phi_rel`, `pt_rel`, `mask`]
    '''
    
    def __init__(self, 
                 dir_path: str='data/', 
                 num_constituents: int=150,
                 clip_neg_pt: bool=False):
        
        self.path = dir_path
        self.num_consts = num_constituents
        self.clip_neg_pt = clip_neg_pt
        self.data_list = self.load_data_list()
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        filename, key = self.data_list[idx]
        data = torch.Tensor(self.load_file(filename, key))

        if len(data.shape)==3:
            N, P, D = data.shape
            
            #...fill with zero padding up to 'num_constituents'

            if D < 3: raise ValueError('Data has wrong shape: {}'.format(data.shape))
            if P > self.num_consts: raise ValueError('Data has wrong shape: {}'.format(data.shape))
            elif P < self.num_consts: 
                zero_rows = torch.zeros(N, self.num_consts - P, D)
                data = torch.cat((data, zero_rows), dim=1) 

            #...get particle features and add mask

            eta_rel = data[..., 0, None]
            phi_rel = data[..., 1, None]
            pt_rel = data[..., 2, None] 
            mask = (data[..., 0] + data[..., 1] + data[..., 2] != 0).int().unsqueeze(-1) 
            data = torch.cat((eta_rel, phi_rel, pt_rel, mask), dim=-1)  

            #...pt-order particles  
        
            _ , i = torch.sort(torch.abs(data[:, :, 2]), dim=1, descending=True) 
            data = torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 

            #...clip negative pt values

            if self.clip_neg_pt:
                data_clip = torch.clone(data)    
                data = torch.zeros_like(data)
                data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
                _ , i = torch.sort(torch.abs(data[:, :, 2]), dim=1, descending=True) 
                data = torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 

        return data
    
        
    def load_file(self, filename, key):
        if filename.endswith('.npy'):
            return np.load(filename)
        elif filename.endswith('.hdf5'):
            with h5py.File(filename, 'r') as f:
                return np.array(f[key])
        elif filename.endswith('.h5'):
            with h5py.File(filename, 'r') as f:
                return np.array(f[key])
        else:
            raise ValueError('No such file format: {}'.format(filename))
    
        
    def load_data_list(self):
        data_list = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.npy'):
                    data_list.append((os.path.join(root, file), None))
                elif file.endswith('.hdf5'):
                    with h5py.File(os.path.join(root, file), 'r') as f:
                        for key in f.keys():
                            print(key)
                            data_list.append((os.path.join(root, file), key))
                elif file.endswith('.h5'):
                    with h5py.File(os.path.join(root, file), 'r') as f:
                        for key in f.keys():
                            print(key)
                            data_list.append((os.path.join(root, file), key))
        return data_list
    
