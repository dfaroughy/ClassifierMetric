import torch
import numpy as np
import os
import h5py
import json
import glob

from torch.utils.data import Dataset

# TODO fix preprocessing via method names

class JetNetDataset(Dataset):

    ''' Arguments:
        - `dir_path` : path to data files
        - `data_files` : dictionary of data files to load, default is `None`
        - `class_labels` : dictionary of class labels for each data file, default is `None`
        - `num_jets`: number of jets to load, default is `None`
        - `num_constituents`: number of particle constituents in each jet, default is `150`
        - `particle_features`: list of particle features to include in data, default is `['eta_rel', 'phi_rel', 'pt_rel']`
        - `preprocess`: TODO preprocessing methods to apply to data, default is `None`
        - `remove_negative_pt`: remove negative pt constituents from jet, default is `False` 
        Loads and formats data from data files in format `.hdf5`.\\
        Adds mask and zero-padding.\\
        Adds class label for each sample via `class_labels`.\\
        Input data in files should always have shape (`#jets`, `#particles`, `#features`) with feaures: `(eta_rel, phi_rel, pt_rel, ...)`
    '''
    
    def __init__(self, 
                 dir_path: str=None, 
                 datasets: dict=None,
                 class_labels: dict=None,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 preprocess : list=['standardize'],
                 num_jets: int=None,
                 num_constituents: int=150,
                 remove_negative_pt: bool=False,
                 compute_jet_features: bool=False):
        
        self.path = dir_path
        self.datasets = datasets
        self.class_labels = class_labels
        self.num_jets = num_jets
        self.num_consts = num_constituents
        self.particle_features = particle_features
        self.remove_negative_pt = remove_negative_pt
        self.compute_jet_features = compute_jet_features
        self.preprocess = preprocess 
        self.summary_statistics = {}
        self.dataset_list = self.get_data()
    
    def __getitem__(self, idx):
        output = {}
        datasets, labels = self.dataset_list
        dataset = datasets[idx]  
        
        if self.compute_jet_features:
            jet = self.get_jet_features(dataset)
            output['jet_features'] = jet

        if self.preprocess:
            dataset = self.apply_preprocessing(dataset, stats=self.summary_statistics['all'])  

        output['label'] = labels[idx]
        output['particle_features'] = dataset[:, :-1]
        output['mask'] = dataset[:, -1]
        return output

    def __len__(self):
        return self.dataset_list[0].size(0)

    def get_data(self):
        print("INFO: loading and preprocessing data from {}".format(self.path))
        data_list = []
        label_list = []
        for data in list(self.datasets.keys()):
            file_name = self.datasets[data][0]
            key = self.datasets[data][1] if len(self.datasets[data]) > 1 else None
            file_path = os.path.join(self.path, file_name)
            if data == 'test' and self.class_labels[data] != -1:
                raise ValueError('Test dataset must have label = -1') 
            with h5py.File(file_path, 'r') as f:
                label = self.class_labels[data] if self.class_labels is not None else None
                print('\t- {} ({}) label: {}'.format(file_name, key, label))
                dataset = torch.from_numpy(f[key][...])
                dataset = self.apply_formatting(dataset)
                self.summary_statistics[label] = self.summary_stats(dataset)
                data_list.append(dataset)
                label_list.append(torch.full((dataset.shape[0],), label))
        data_tensor = torch.cat(data_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0) 
        self.summary_statistics['all'] = self.summary_stats(data_tensor)
        return data_tensor, label_tensor

    def apply_formatting(self, sample):
        sample = FormatData(sample,
                            num_jets=self.num_jets,
                            num_constituents=self.num_consts,
                            particle_features=self.particle_features,
                            remove_negative_pt=self.remove_negative_pt)
        sample.format()
        return sample.data
    
    def apply_preprocessing(self, sample, stats):
        sample = PreprocessData(data=sample, stats=stats)
        sample.standardize()
        return sample.jet
    
    def get_jet_features(self, sample):
        sample = PreprocessData(data=sample)
        return sample.jet_features
    
    def summary_stats(self, data):
        data_flat = data.view(-1, data.shape[-1])
        mask = data_flat[:, -1].bool()
        mean = torch.mean(data_flat[mask],dim=0)
        std = torch.std(data_flat[mask],dim=0)
        min, _ = torch.min(data_flat[mask],dim=0)
        max, _ = torch.max(data_flat[mask],dim=0)
        mean, std, min, max = mean[:-1], std[:-1], min[:-1], max[:-1]
        return (mean, std, min, max)
    
    def save(self, path):
        print("INFO: saving dataset to {}".format(path))
        torch.save(self.dataset_list, os.path.join(path, 'dataset.pth'))
        dataset_args = {'dir_path': self.path, 
                     'datasets': self.datasets, 
                     'class_labels': self.class_labels, 
                     'particle_features': self.particle_features,
                     'preprocess': self.preprocess, 
                     'num_jets': self.num_jets, 
                     'num_constituents': self.num_consts, 
                     'remove_negative_pt': self.remove_negative_pt}
        with open(path+'/dataset_configs.json', 'w') as json_file:
            json.dump(dataset_args, json_file, indent=4)

    @staticmethod
    def load(path):
        print("INFO: loading dataset from {}".format(path))
        dataset_list = torch.load(os.path.join(path, 'dataset.pth'))
        with open(path+'/dataset_configs.json') as json_file:
            dataset_args = json.load(json_file)
            loaded_dataset = JetNetDataset(**dataset_args)
        loaded_dataset.dataset_list = dataset_list
        return loaded_dataset

class FormatData:

    ''' This module formats the data in the following way:
        - zero padding
        - extract particle constituents features e.g. eta_rel, phi_rel, pt_rel, log_pt_rel, R, etc...
        - removes particles with negative pt (optional)
        - orders particles constituents by pt (descending)
        - trims dataset to desired number of jets and constituents
    '''

    def __init__(self, 
                 data: torch.Tensor=None,
                 num_jets: int=None,
                 num_constituents: int=150,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 remove_negative_pt: bool=False
                ):
        
        self.data = data
        self.num_jets = data.shape[0] if num_jets is None else num_jets
        self.num_consts = num_constituents
        self.particle_features = particle_features
        self.remove_negative_pt = remove_negative_pt

    def data_rank(self, n):
        return len(self.data.shape) == n

    def format(self):
        if self.data_rank(3):  
            self.zero_padding()
            self.get_particle_features()
            if self.remove_negative_pt: self.remove_neg_pt() 
            self.trim_dataset()
        if self.data_rank(2): 
            # TODO
            pass
    
    def zero_padding(self):
        N, P, D = self.data.shape
        if P < self.num_consts:
            zero_rows = torch.zeros(N, self.num_consts - P, D)
            self.data =  torch.cat((self.data, zero_rows), dim=1)
        else: pass 

    def get_particle_features(self, masked: bool=True): 
        pf = {}
        pf['eta_rel'] = self.data[..., 0, None]
        pf['phi_rel'] = self.data[..., 1, None]
        pf['pt_rel'] = self.data[..., 2, None]
        pf['e_rel'] = pf['pt_rel'] * torch.cosh(pf['eta_rel'])
        pf['log_pt_rel'] = torch.log(pf['pt_rel'])
        pf['log_e_rel'] = torch.log(pf['e_rel'])
        pf['R'] = torch.sqrt(pf['eta_rel']**2 + pf['phi_rel']**2)
        features = [pf[f] for f in self.particle_features]
        if masked:
            mask = (self.data[..., 2] != 0).int().unsqueeze(-1) 
            features += [mask]
        self.data = torch.cat(features, dim=-1)
        self.pt_order()        

    def remove_neg_pt(self):
        data_clip = torch.clone(self.data)    
        self.data = torch.zeros_like(self.data)
        self.data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
        self.pt_order()
    
    def trim_dataset(self):
        if self.data_rank(3): 
            self.data = self.data[:self.num_jets, :self.num_consts, :]
        if self.data_rank(2): 
            self.data = self.data[:self.num_conts, :] 

    def pt_order(self):
        for i, f in enumerate(self.particle_features):
            if 'pt' in f: 
                idx = i
                break
        if self.data_rank(3): 
            _ , i = torch.sort(torch.abs(self.data[:, :, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 
        if self.data_rank(2):  
            _ , i = torch.sort(torch.abs(self.data[:, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 

class PreprocessData:

    ''' Module for preprocessing jets. 
        args:
        - `data`: torch tensor of jet constituents
        - `stats`: tuple of summary statistics of the dataset (`mean`, `std`, `min`, `max`)

        preprocessing options are the following:
        - center jets
        - standardize
        - normalize
        - logit transform
        
       The method `get_jet_features` computes from the particle constituents the main jet-level features (pt, eta, phi, mass, multuiplicity)
    '''

    def __init__(self, 
                 data: torch.Tensor=None, 
                 stats: tuple=None):
        
        self.jet = data
        self.dim_features = self.jet.shape[-1] - 1
        if stats is not None:
            self.mean, self.std, self.min, self.max = stats 
        self.mask = self.jet[:, -1, None]
        self.jet_unmask = self.jet[:, :self.dim_features]
    
    def get_jet_features(self):
        # slow
        mask = self.mask.squeeze(-1)
        eta, phi, pt = self.jet[:, 0], self.jet[:, 1], self.jet[:, 2]
        multiplicity = torch.sum(mask, dim=0)
        e_j  = torch.sum(mask * pt * torch.cosh(eta), dim=0)
        px_j = torch.sum(mask * pt * torch.cos(phi), dim=0)
        py_j = torch.sum(mask * pt * torch.sin(phi), dim=0)
        pz_j = torch.sum(mask * pt * torch.sinh(eta), dim=0)
        pt_j = torch.sqrt(px_j**2 + py_j**2)
        m_j  = torch.sqrt(e_j**2 - px_j**2 - py_j**2 - pz_j**2)
        eta_j = torch.asinh(pz_j / pt_j)
        phi_j = torch.atan2(py_j, px_j)
        self.jet_features = torch.Tensor((pt_j, eta_j, phi_j, m_j, multiplicity))
        
    def center_jets(self):
        # slow
        N = self.jet.shape[0]
        jet_coords = self.jet_features[1:3] # jet (eta, phi)
        jet_coords = jet_coords.repeat(N, 1) * self.mask
        zeros = torch.zeros((N, self.dim_features - 2))
        jet_coords = torch.cat((jet_coords, zeros), dim=1)
        self.jet_unmask -= jet_coords 
        self.jet_unmask -= jet_coords 
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)

    def standardize(self,  sigma: float=1.0):
        self.jet_unmask = (self.jet_unmask * self.mean) * (1e-8 + sigma / self.std )
        self.jet_unmask = self.jet_unmask * self.mask
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)

    def normalize(self):
        self.jet_unmask = (self.jet_unmask - self.min) / ( self.max - self.min )
        self.jet_unmask = self.jet_unmask * self.mask
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)
    
    def logit_tramsform(self, alpha=1e-6):
        self.jet_unmask = self.logit(self.jet_unmask, alpha=alpha)
        self.jet_unmask = self.jet_unmask * self.mask
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)

    def logit(t, alpha=1e-6):
        x = alpha + (1 - 2 * alpha) * t
        return torch.log(x/(1-x))