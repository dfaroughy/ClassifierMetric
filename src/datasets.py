import torch
import numpy as np
import os
import h5py
from tabulate import tabulate
from torch.utils.data import Dataset
from src.jetnet import JetNetFeatures


class JetNetDataSets(Dataset):

    ''' Arguments:
        - `dir_path` : path to data files
        - `data_files` : dictionary of data files to load, default is `None`
        - `data_class_labels` : dictionary of class labels for each data file, default is `None`
        - `num_jets`: number of jets to load, default is `None`
        - `num_constituents`: number of particle constituents in each jet, default is `150`
        - `particle_features`: list of particle features to include in data, default is `['eta_rel', 'phi_rel', 'pt_rel']`
        - `preprocess`: dictionary of preprocessing methods to apply to data, default is `None`
        - `clip_neg_pt`: clip negative pt values to zero, default is `False`
    
        Loads and formats data from data files in format `.npy` or `.hdf5`.\\
        Adds mask and zero-padding if necessary.\\
        Adds class label for each sample if `data_class_labels` is provided.\\
        Input data in files should always have shape (`#jets`, `#particles`, `#features`) with feaures: `(eta_rel, phi_rel, pt_rel, ...)`
        
    '''
    
    def __init__(self, 
                 dir_path: str=None, 
                 data_files: dict=None,
                 data_class_labels: dict=None,
                 num_jets: int=None,
                 num_constituents: int=150,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 preprocess: dict=None,
                 clip_neg_pt: bool=False):
        
        self.path = dir_path
        self.data_files = data_files
        self.data_class_labels = data_class_labels
        self.num_jets = num_jets
        self.num_consts = num_constituents
        self.particle_features = particle_features
        self.preprocess = preprocess 
        self.clip_neg_pt = clip_neg_pt
        self.jet_list = self.dataloader()
 
    #...data formatting methods

    def format_data(self, data):
        data = self.zero_padding(data)
        data = self.get_particle_features(data)
        data = self.clip_negative_pt(data) 
        data = self.remove_soft_particles(data)
        return data

    def zero_padding(self, data):
        N, P, D = data.shape
        if P < self.num_consts:
            zero_rows = torch.zeros(N, self.num_consts - P, D)
            return torch.cat((data, zero_rows), dim=1) 
        else: 
            return data
        
    def get_particle_features(self, data, masked=True): 
        pf = {}
        pf['eta_rel'] = data[..., 0, None]
        pf['phi_rel'] = data[..., 1, None]
        pf['pt_rel'] = data[..., 2, None]
        pf['R'] = torch.sqrt(pf['eta_rel']**2 + pf['phi_rel']**2)
        pf['e_rel'] = pf['pt_rel'] * torch.cosh(pf['eta_rel'])
        features = [pf[f] for f in self.particle_features]
        if masked:
            mask = (data[..., 0] + data[..., 1] + data[..., 2] != 0).int().unsqueeze(-1) 
            features += [mask]
        data = torch.cat(features, dim=-1)          
        return self.pt_order(data)
    
    def clip_negative_pt(self, data):
        data_clip = torch.clone(data)    
        data = torch.zeros_like(data)
        data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
        return self.pt_order(data) if self.clip_neg_pt else data
    
    def remove_soft_particles(self, data):
        _, P, _ = data.shape
        return  data[:, :self.num_consts, :] if P > self.num_consts else data

    def pt_order(self, data):
        _ , i = torch.sort(torch.abs(data[:, :, 2]), dim=1, descending=True) 
        return torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 
        
    #...dataset loading methods

    def __len__(self):
        return self.jets[0].size(0)
    
    def __getitem__(self, idx):

        jet, labels = self.jet_list[idx]
        if self.preprocess is not None: 
            # TODO implement jet-level preprocessing
            # mean, std, min, max are available at this point!
            pass

        return jet, labels

    def dataloader(self):

        samples = []
        labels = []

        for root, dirs, files in os.walk(self.path):
            for file in files:
                key = None
                path = os.path.join(root, file)
                # hdf5 files
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    with h5py.File(path, 'r') as f:
                        for key in f.keys():
                            if self.data_files is not None:
                                for k in self.data_files.keys():
                                    if file in self.data_files[k][0] and key==self.data_files[k][1]:
                                        label = None if self.data_class_labels is None else self.data_class_labels[k]
                                        data = torch.Tensor(np.array(f[key]))
                                        data = self.format_data(data)
                                        data = data[:self.num_jets]
                                        samples.append(data)
                                        labels.append(torch.full((data.shape[0],), label))
        
        samples = torch.cat(samples, dim=0)
        labels = torch.cat(labels, dim=0) 
        samples_flat = samples.view(-1, samples.shape[-1])
        mask = samples_flat[:, -1].bool()
        self.mean = torch.mean(samples_flat[mask],dim=0)
        self.std = torch.std(samples_flat[mask],dim=0)
        self.min,_ = torch.min(samples_flat[mask],dim=0)
        self.max,_ = torch.max(samples_flat[mask],dim=0)

        return samples, labels
    
    def data_summary(self):
        table = []
        headers = ['Sample Name', 'Filename', 'Extension', 'Key', 'Shape', 'Class Label']
        for name, filename, key, label in self.jet_list:
            data = self.load_file(filename, key)
            filename, ext = os.path.splitext(filename)
            filename = os.path.basename(filename)
            table.append([name, filename, ext, key, str(data.shape), label])

        print(tabulate(table, headers=headers, tablefmt='pretty'))  

