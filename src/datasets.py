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
        self.data_list = self.load_data_list()
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        name, filename, key, label = self.data_list[idx]
        data = torch.Tensor(self.load_file(filename, key))

        N = data.shape[0]

        if len(data.shape) == 3:

            N, P, D = data.shape
            
            #...fill with zero padding up to 'num_constituents'

            if D < 3: raise ValueError('Data has wrong shape: {}'.format(data.shape))
            elif P < self.num_consts: 
                zero_rows = torch.zeros(N, self.num_consts - P, D)
                data = torch.cat((data, zero_rows), dim=1) 

            #...get particle features

            pf = {}
            pf['eta_rel'] = data[..., 0, None]
            pf['phi_rel'] = data[..., 1, None]
            pf['pt_rel'] = data[..., 2, None]
            pf['R'] = torch.sqrt(pf['eta_rel']**2 + pf['phi_rel']**2)
            pf['e_rel'] = pf['pt_rel'] * torch.cosh(pf['eta_rel'])
            features = [pf[f] for f in self.particle_features]

            #...add mask to indicate which particles are padded

            mask = (data[..., 0] + data[..., 1] + data[..., 2] != 0).int().unsqueeze(-1) 
            features += [mask]
            data = torch.cat(features, dim=-1)          
            data = self._pt_order(data)

            #...clip negative pt values

            if self.clip_neg_pt:
                data_clip = torch.clone(data)    
                data = torch.zeros_like(data)
                data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
                data = self._pt_order(data)

            #...preprocess data

            if self.preprocess is not None:
                _data = JetNetFeatures(data)
                _data.preprocess(methods=self.preprocess, name=name)
                data = _data.particles

            #...remove softest jet constituents if more than 'num_constituents'

            if P > self.num_consts:
                data = data[:, :self.num_consts, :]

        if label is None: 
            return data[:self.num_jets if self.num_jets is not None else N]
        else:
            data = data[:self.num_jets if self.num_jets is not None else N] 
            labels = torch.full_like(data[..., 0], label).unsqueeze(-1)
            return data, labels
    
    def _pt_order(self, data):
        _ , i = torch.sort(torch.abs(data[:, :, 2]), dim=1, descending=True) 
        return torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 
        
    def load_file(self, filename, key):
        filename = os.path.join(self.path, filename)
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
            i = 0
            for file in files:
                key = None
                path = os.path.join(root, file)

                # numpy files
                if file.endswith('.npy'):
                    i+=1
                    if self.data_files is not None:
                        for k in self.data_files.keys():
                            if file in self.data_files[k][0] and key==self.data_files[k][1]:
                                label = None if self.data_class_labels is None else self.data_class_labels[k]
                                data_list.append((k, file, key, label))  
                    else:
                        label = None if self.data_class_labels is None else i
                        data_list.append(('sample_{}'.format(i), file, key, label))   

                # hdf5 files
                elif file.endswith('.hdf5'):
                    with h5py.File(path, 'r') as f:
                        for key in f.keys():
                            i+=1
                            if self.data_files is not None:
                                for k in self.data_files.keys():
                                    if file in self.data_files[k][0] and key==self.data_files[k][1]:
                                        label = None if self.data_class_labels is None else self.data_class_labels[k]
                                        data_list.append((k, file, key, label))
                            else:
                                label = None if self.data_class_labels is None else i
                                data_list.append(('sample_{}'.format(i), file, key, label))

                elif file.endswith('.h5'):
                    with h5py.File(path, 'r') as f:
                        for key in f.keys():
                            i+=1
                            if self.data_files is not None:
                                for k in self.data_files.keys():
                                    if file in self.data_files[k][0] and key==self.data_files[k][1]:
                                        label = None if self.data_class_labels is None else self.data_class_labels[k]
                                        data_list.append((k, file, key, label))
                            else:
                                label = None if self.data_class_labels is None else i
                                data_list.append(('sample_{}'.format(i), file, key, label))
        return data_list
    
    def data_summary(self):
        table = []
        headers = ['Sample Name', 'Filename', 'Extension', 'Key', 'Shape', 'Class Label']
        for name, filename, key, label in self.data_list:
            data = self.load_file(filename, key)
            filename, ext = os.path.splitext(filename)
            filename = os.path.basename(filename)
            table.append([name, filename, ext, key, str(data.shape), label])

        print(tabulate(table, headers=headers, tablefmt='pretty'))  



