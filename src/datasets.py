import torch
import numpy as np
import os
import h5py
from tabulate import tabulate
from torch.utils.data import Dataset
from src.jetnet import JetNetFeatures



class JetNetDataset(Dataset):

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
                 preprocess : bool=False,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 clip_negative_pt: bool=False):
        
        self.path = dir_path
        self.data_files = data_files
        self.data_class_labels = data_class_labels
        self.num_jets = num_jets
        self.num_consts = num_constituents
        self.particle_features = particle_features
        self.clip_negative_pt = clip_negative_pt
        self.preprocess = preprocess 
        self.dataset_list = self.get_data()
 
    def __len__(self):
        return self.dataset_list[0].size(0)
    
    def __getitem__(self, idx):
        datasets, labels, stats = self.dataset_list
        if self.preprocess:
            datasets = self.apply_preprocessing(datasets[idx], stats=stats[idx])  
        return datasets[idx], labels[idx]
         
    def get_data(self):
        data_list = []
        label_list = []
        stats_list=[]
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
                                        dataset = torch.Tensor(np.array(f[key]))
                                        dataset = self.apply_formatting(dataset) 
                                        data_list.append(dataset)
                                        stats_list.append(self.summary_stat(dataset).repeat(dataset.shape[0], 1))
                                        label_list.append(torch.full((dataset.shape[0],), label))
                
                # elif file.endswith('.npy'):
                #     # TODO 
                #     raise NotImplementedError('Numpy files not supported yet')

        data_tensor = torch.cat(data_list, dim=0)
        stats_tensor = torch.cat(stats_list, dim=0) 
        label_tensor = torch.cat(label_list, dim=0) 
        return data_tensor, label_tensor, stats_tensor
    
    def summary_stats(self, data):
        ''' Returns mean, std, min, max of data'''
        data_flat = data.view(-1, data.shape[-1])
        mask = data_flat[:, -1].bool()
        mean = torch.mean(data_flat[mask],dim=0)
        std = torch.std(data_flat[mask],dim=0)
        min,_ = torch.min(data_flat[mask],dim=0)
        max,_ = torch.max(data_flat[mask],dim=0)
        return torch.cat((mean, std, min, max), dim=-1)

    def apply_formatting(self, sample):
        sample = FormatData(sample,
                          num_jets=self.num_jets,
                          num_constituents=self.num_consts,
                          particle_features=self.particle_features,
                          clip_negative_pt=self.clip_negative_pt)
        sample.format()
        return sample.data
    
    def apply_preprocessing(self, sample, stats):
        sample = PreprocessData(data=sample, stats=stats)
        sample.standardize()    
        return sample.jet


class FormatData:
    def __init__(self, 
                 data: torch.Tensor=None,
                 num_jets: int=None,
                 num_constituents: int=None,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 clip_negative_pt: bool=False
                ):
        
        self.data = data
        self.num_jets = num_jets
        self.num_consts = num_constituents
        self.particle_features = particle_features
        self.clip_negative_pt = clip_negative_pt

    def data_rank(self, n):
        return len(self.data.shape) == n

    def format(self):
        if self.data_rank(3):  
            self.zero_padding()
            self.get_particle_features()
            self.clip_neg_pt() 
        if self.data_rank(2): 
            # TODO
            pass
        self.trim_dataset()
    
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

    def clip_neg_pt(self):
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
    
    def __init__(self, 
                 data: torch.Tensor=None, 
                 info: torch.Tensor=None):
        
        self.jet = data
        self.dim_features = self.jet.shape[-1] - 1
        info = torch.reshape(info, (4, -1))
        self.mean, self.std, self.max, self.min = tuple(info[i][:self.dim_features] for i in range(info.shape[0]))
        self.mask = self.jet[:, -1, None]
        self.jet_unmask = self.jet[:, :self.dim_features]
        # self.jet_features = self.get_jet_features()

    def get_jet_features(self):
        eta, phi, pt = self.jet=t[:, 0], self.jet[:, 1], self.jet[:, 2]
        multiplicity = torch.sum(self.mask, dim=1)
        e_j  = torch.sum(self.mask * pt * torch.cosh(eta), dim=1)
        px_j = torch.sum(self.mask * pt * torch.cos(phi), dim=1)
        py_j = torch.sum(self.mask * pt * torch.sin(phi), dim=1)
        pz_j = torch.sum(self.mask * pt * torch.sinh(eta), dim=1)
        pt_j = torch.sqrt(px_j**2 + py_j**2)
        m_j  = torch.sqrt(e_j**2 - px_j**2 - py_j**2 - pz_j**2)
        eta_j = torch.asinh(pz_j / pt_j)
        phi_j = torch.atan2(py_j, px_j)
        return torch.Tensor((pt_j, eta_j, phi_j, m_j, multiplicity))
        
    def center_jets(data):
        # TODO
        # data = data[:, :, [2, 0, 1]]
        # etas = jet_etas(data)
        # phis = jet_phis(data)
        # etas = etas[:, np.newaxis].repeat(repeats=data.shape[1], axis=1)
        # phis = phis[:, np.newaxis].repeat(repeats=data.shape[1], axis=1)
        # mask = data[..., 0] > 0  # mask all particles with nonzero pt
        # data[mask, 1] -= etas[mask]
        # data[mask, 2] -= phis[mask]
        # return data[:, :, [1, 2, 0]]
        pass

    def standardize(self,  sigma: float=1.0):
        self.jet_unmask = (self.jet_unmask * self.mean) * (1e-8 + sigma / self.std )
        self.jet_unmask = self.jet_unmask * self.mask
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)

    def normalize(self):
        print(1, self.jet_unmask.shape)

        self.jet_unmask = (self.jet_unmask - self.min) / ( self.max - self.min )
        
        print(2, self.jet_unmask.shape)

        self.jet_unmask = self.jet_unmask * self.mask
        print(3, self.jet_unmask.shape)
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)
    
    def logit_tramsform(self, alpha=1e-6):
        self.jet_unmask = self.logit(self.jet_unmask, alpha=alpha)
        self.jet_unmask = self.jet_unmask * self.mask
        self.jet = torch.cat((self.jet_unmask, self.mask), dim=-1)

    def logit(t, alpha=1e-6):
        x = alpha + (1 - 2 * alpha) * t
        return torch.log(x/(1-x))
    



# class JetNetDataset_Legacy(Dataset):

#     ''' Arguments:
#         - `dir_path` : path to data files
#         - `data_files` : dictionary of data files to load, default is `None`
#         - `data_class_labels` : dictionary of class labels for each data file, default is `None`
#         - `num_jets`: number of jets to load, default is `None`
#         - `num_constituents`: number of particle constituents in each jet, default is `150`
#         - `particle_features`: list of particle features to include in data, default is `['eta_rel', 'phi_rel', 'pt_rel']`
#         - `preprocess`: dictionary of preprocessing methods to apply to data, default is `None`
#         - `clip_neg_pt`: clip negative pt values to zero, default is `False`
    
#         Loads and formats data from data files in format `.npy` or `.hdf5`.\\
#         Adds mask and zero-padding if necessary.\\
#         Adds class label for each sample if `data_class_labels` is provided.\\
#         Input data in files should always have shape (`#jets`, `#particles`, `#features`) with feaures: `(eta_rel, phi_rel, pt_rel, ...)`
        
#     '''
    
#     def __init__(self, 
#                  dir_path: str=None, 
#                  data_files: dict=None,
#                  data_class_labels: dict=None,
#                  num_jets: int=None,
#                  num_constituents: int=150,
#                  preprocess : bool=False,
#                  particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
#                  clip_neg_pt: bool=False):
        
#         self.path = dir_path
#         self.data_files = data_files
#         self.data_class_labels = data_class_labels
#         self.num_jets = num_jets
#         self.num_consts = num_constituents
#         self.particle_features = particle_features
#         self.clip_neg_pt = clip_neg_pt
#         self.preprocess = preprocess 
#         self.dataset_list = self.dataloader()
 
#     #...data formatting methods

#     def format_data(self, data):
#         data = self.zero_padding(data)
#         data = self.get_particle_features(data)
#         data = self.clip_negative_pt(data) 
#         data = self.remove_soft_particles(data)
#         return data

#     def zero_padding(self, data):
#         N, P, D = data.shape
#         if P < self.num_consts:
#             zero_rows = torch.zeros(N, self.num_consts - P, D)
#             return torch.cat((data, zero_rows), dim=1) 
#         else: 
#             return data
        
#     def get_particle_features(self, data, masked=True): 
#         pf = {}
#         pf['eta_rel'] = data[..., 0, None]
#         pf['phi_rel'] = data[..., 1, None]
#         pf['pt_rel'] = data[..., 2, None]
#         pf['e_rel'] = pf['pt_rel'] * torch.cosh(pf['eta_rel'])
#         pf['log_pt_rel'] = torch.log(pf['pt_rel'])
#         pf['log_e_rel'] = torch.log(pf['e_rel'])
#         pf['R'] = torch.sqrt(pf['eta_rel']**2 + pf['phi_rel']**2)

#         features = [pf[f] for f in self.particle_features]
#         if masked:
#             mask = (data[..., 0] + data[..., 1] + data[..., 2] != 0).int().unsqueeze(-1) 
#             features += [mask]
#         data = torch.cat(features, dim=-1)          
#         return self.pt_order(data)
    
#     def clip_negative_pt(self, data):
#         data_clip = torch.clone(data)    
#         data = torch.zeros_like(data)
#         data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
#         return self.pt_order(data) if self.clip_neg_pt else data
    
#     def remove_soft_particles(self, data):
#         _, P, _ = data.shape
#         return  data[:, :self.num_consts, :] if P > self.num_consts else data

#     def pt_order(self, data):
#         _ , i = torch.sort(torch.abs(data[:, :, 2]), dim=1, descending=True) 
#         return torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 
        
#     #...dataset loading methods

#     def __len__(self):
#         return self.dataset_list[0].size(0)
    
#     def __getitem__(self, idx):

#         jets, labels = self.dataset_list

#         if self.preprocess:
#             jets = JetNetPreprocess(jets)

#             #...get jets 

#             particle_feats.logit_tramsform()
#             particle_feats.standardize() 

#             #...get jets 

#             jets.normalize()
#             jets.logit_tramsform()
#             jets.standardize()    

#         return jet_feat, jets[idx], labels[idx]

#     def dataloader(self):
#         particle_feats = []
#         labels = []
#         for root, dirs, files in os.walk(self.path):
#             for file in files:
#                 key = None
#                 path = os.path.join(root, file)

#                 # hdf5 files
#                 if file.endswith('.hdf5') or file.endswith('.h5'):
#                     with h5py.File(path, 'r') as f:
#                         for key in f.keys():
#                             if self.data_files is not None:
#                                 for k in self.data_files.keys():
#                                     if file in self.data_files[k][0] and key==self.data_files[k][1]:
#                                         label = None if self.data_class_labels is None else self.data_class_labels[k]
#                                         data = torch.Tensor(np.array(f[key]))
#                                         data = self.format_data(data)
#                                         data = data[:self.num_jets]
#                                         particle_feats.append(data)
#                                         labels.append(torch.full((data.shape[0],), label))
#                 # elif file.endswith('.npy'):
#                 #     #TODO add support for numpy files
#                 #     raise NotImplementedError('Numpy files not supported yet.')

#         particle_feats = torch.cat(particle_feats, dim=0)
#         labels = torch.cat(labels, dim=0) 
#         particle_feats_flat = particle_feats.view(-1, particle_feats.shape[-1])
#         mask = particle_feats_flat[:, -1].bool()
#         sample_mean = torch.mean(particle_feats_flat[mask],dim=0)
#         sample_std = torch.std(particle_feats_flat[mask],dim=0)
#         sample_min,_ = torch.min(particle_feats_flat[mask],dim=0)
#         sample_max,_ = torch.max(particle_feats_flat[mask],dim=0)
#         self.info = (sample_mean, sample_std, sample_min, sample_max)
#         return particle_feats, labels

