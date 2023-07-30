import torch
import os
import h5py
import json
from torch.utils.data import Dataset
from src.preprocess import PreprocessData

class JetNetDataset(Dataset):

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
        self.preprocess_methods = preprocess 
        self.summary_statistics = {}
        self.dataset_list = self.get_data()

    def __getitem__(self, idx):
        output = {}
        datasets, labels = self.dataset_list
        output['label'] = labels[idx]
        output['mask'] = datasets[idx][:, -1]
        particles, jet = self.apply_preprocessing(sample=datasets[idx])  
        output['particle_features'] = particles
        if jet is not None: output['jet_features'] = jet 
        return output

    def __len__(self):
        return self.dataset_list[0].size(0)

    def get_data(self):
        print("INFO: loading and preprocessing data from {}".format(self.path))
        data_list, label_list = [], []
        for data in list(self.datasets.keys()):
            file_name = self.datasets[data][0]
            key = self.datasets[data][1] if len(self.datasets[data]) > 1 else None
            file_path = os.path.join(self.path, file_name)
            with h5py.File(file_path, 'r') as f:
                label = self.class_labels[data] if self.class_labels is not None else None
                print('\t- {}, file: {}, key: {}, label: {} {}'.format(data, file_name, key, label, '(test)' if label==-1 else ''))
                dataset = torch.from_numpy(f[key][...])
                dataset = self.apply_formatting(dataset)
                self.summary_statistics[label] = self.summary_stats(dataset)
                data_list.append(dataset)
                label_list.append(torch.full((dataset.shape[0],), label))
        data_tensor = torch.cat(data_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0) 
        self.summary_statistics['dataset'] = self.summary_stats(data_tensor)
        return data_tensor, label_tensor

    def apply_formatting(self, sample):
        sample = FormatData(sample,
                            num_jets=self.num_jets,
                            num_constituents=self.num_consts,
                            particle_features=self.particle_features,
                            remove_negative_pt=self.remove_negative_pt)
        sample.format()
        return sample.data
    
    def apply_preprocessing(self, sample):
        sample = PreprocessData(data=sample, 
                                stats=self.summary_statistics['dataset'],
                                methods = self.preprocess_methods, 
                                compute_jet_features=self.compute_jet_features)
        sample.preprocess()
        return sample.particle_features, sample.jet_features
    
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
