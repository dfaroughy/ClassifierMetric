import numpy as np
import torch
from torch import Tensor
import seaborn as sns
import matplotlib.pyplot as plt

'''
class for getting particle and jet features from jetnet-style data. 
    - input events shape: (num_jets, num_consts, dim)
    - features: (eta_rel, phi_rel, pt_rel)    
attributes:
    - particles features (eta_rel, phi_rel, pt_rel, e_rel, R)
methods:
    - get_jet_features: get jet features from constituent features
    - get_mean_std: get mean and std of constituent features
    - nth_particle: get nth particle feature
    - pt_smear (distort): smear pt of constituents
    - coordinate_shift (distort): shift eta and phi of constituents
    - standardize (preprocess): standardize constituent features
    - normalize (preprocess): normalize constituent features
    - logit_transform (preprocess): apply logit transform to constituent features
    - preprocess: apply preprocessing methods
    - postprocess: apply postprocessing methods
    - particle_plot: plot particle features
    - jet_plot: plot jet features
    - image: plot image of jet (for events and average events)
'''

class JetNetFeatures:
    
    def __init__(self, 
                 events: torch.Tensor=None, 
                 masked: bool=False, 
                 max_num_consts: int=150): 
    
        self.max_num_consts = max_num_consts
        self.num_jets, num_consts, dim = events.shape
        
        # fill with zero padding up to 'max_num_consts' if necessary

        if num_consts < max_num_consts:
            zero_rows = torch.zeros(self.num_jets, max_num_consts - num_consts, dim)
            events = torch.cat((events, zero_rows), dim=1) 

        # get particle features

        eta_rel = events[..., 0, None]
        phi_rel = events[..., 1, None]
        pt_rel = events[..., 2, None] 
        mask = events[..., 3, None] if masked else (events[..., 0] + events[..., 1] + events[..., 2] != 0).int().unsqueeze(-1) 
        R = torch.sqrt(eta_rel**2 + phi_rel**2)
        e_rel = pt_rel * torch.cosh(eta_rel)
    
        data = torch.cat((eta_rel, phi_rel, pt_rel, e_rel, R, mask), dim=-1)  

        # pt-order particles
        
        _ , i = torch.sort(data[:, :, 2], dim=1, descending=True) 
        particles = torch.gather(data, 1, i.unsqueeze(-1).expand_as(data)) 

        # clip negative momenta

        self.particles = torch.zeros_like(particles)
        self.particles[particles[..., 2] >= 0.0] = particles[particles[..., 2] >= 0.0]
      
    #...default features

    @property 
    def features(self): 
        return self.particles[..., :5] 
    @property 
    def eta_rel(self): 
        return self.particles[..., 0] 
    @property 
    def phi_rel(self): 
        return self.particles[..., 1]
    @property 
    def pt_rel(self): 
        return self.particles[..., 2]
    @property 
    def R(self): 
        return self.particles[..., 3]
    @property 
    def mask(self): 
        return self.particles[..., -1] 
    @property 
    def multiplicity(self):
        return self.particles[..., -1].sum(dim=-1)

    def get_mean_std(self):
        self.mean = torch.mean(self.particles, dim=-1)
        self.std = torch.std(self.particles, dim=-1)

    #...jet features from constituents

    def get_jet_features(self):
        
        pt = self.particles[..., 2, None]
        eta = self.particles[..., 0, None]
        phi = self.particles[..., 1, None]
        mask = self.particles[..., -1, None]
        multiplicity = torch.sum(mask, dim=1)

        e_j  = torch.sum(mask * pt * torch.cosh(eta), dim=1)
        px_j = torch.sum(mask * pt * torch.cos(phi), dim=1)
        py_j = torch.sum(mask * pt * torch.sin(phi), dim=1)
        pz_j = torch.sum(mask * pt * torch.sinh(eta), dim=1)
        pt_j = torch.sqrt(px_j**2 + py_j**2)
        m_j  = torch.sqrt(e_j**2 - px_j**2 - py_j**2 - pz_j**2)
        eta_j = torch.asinh(pz_j / pt_j)
        phi_j = torch.atan2(py_j, px_j)
        
        self.jets = torch.cat((pt_j, eta_j, phi_j, m_j, e_j, multiplicity), dim=-1) 

    def nth_particle(self, n, feature: str=None):
        particles = self.particles[..., n-1, :]
        mask = particles[..., -1] > 0
        if feature is None:
            return particles[mask][..., :-1]
        else:
            idx = {'eta_rel':0, 'phi_rel':1, 'pt_rel':2, 'R':3}
            return particles[mask][..., idx[feature]] 
    
    #...data distortions

    def pt_smear(self, scale: float=0.001):
        mask = self.particles[..., -1] 
        energy_resolution = np.sqrt( (0.00025 * self.particles[..., 2])**2 + 0.015**2) 
        noise = scale * energy_resolution * torch.randn_like(self.particles[..., 2])
        self.particles[..., 2] = self.particles[..., 2] - mask * torch.abs(noise)
  
    def coordinate_shift(self, loc_eta: float=0.01, loc_phi: float=0.0):
        mask = self.particles[..., -1] 
        eta_noise = loc_eta * torch.randn_like(self.particles[..., 0])
        phi_noise = loc_phi * torch.randn_like(self.particles[..., 1])
        self.particles[..., 0] = self.particles[..., 0] + mask * eta_noise
        self.particles[..., 1] = self.particles[..., 1] + mask * phi_noise

    #...data processing

    def log_pt(self, inverse: bool=False):
        mask = self.particles[..., -1].bool()
        if not inverse:
            self.particles[..., 2][mask] = torch.log(self.particles[..., 2][mask]) # log(pt_rel)
            self.particles[..., 3][mask] = torch.log(self.particles[..., 3][mask]) # log(e_rel)
            print('INFO: applying log(pt) transform')
        else:
            self.particles[..., 2][mask] = torch.exp(self.particles[..., 2][mask]) 
            self.particles[..., 3][mask] = torch.exp(self.particles[..., 3][mask])
            print('INFO: applying exp(pt) transform')    

    def standardize(self, inverse: bool=False, sigma: float=1.0):
        mask = self.particles[..., -1].bool()
        if not inverse: 
            self.mean = torch.mean(self.particles[..., :5][mask], dim=0, keepdim=True)
            self.std = torch.std(self.particles[..., :5][mask], dim=0, keepdim=True)
            self.particles[..., :5][mask] = (self.particles[..., :5][mask] * self.mean) * (sigma / self.std )
            print('INFO: standardizing data to zero-mean and std={}'.format(sigma)) 
        else:
            self.particles[..., :5][mask] = self.particles[..., :5][mask] * (self.std / sigma) +  self.mean 
            print('INFO: un-standardizing data')

    def normalize(self, inverse: bool=False):
        mask = self.particles[..., -1].bool()
        if not inverse:
            self.max, _ = torch.max(self.particles[..., :5][mask], dim=0, keepdim=True)
            self.min, _ = torch.min(self.particles[..., :5][mask], dim=0, keepdim=True)
            self.particles[..., :5][mask] = (self.particles[..., :5][mask] - self.min) / (self.max - self.min) 
            print('INFO: normalizing data')
        else:
            self.particles[..., :5][mask] = self.particles[..., :5][mask] * (self.max - self.min) + self.min * mask 
            print('INFO: un-normalizing data')

    def logit_transform(self, alpha: float=1e-6, inverse: bool=False):
        mask = self.particles[..., -1].bool()
        if not inverse:
            self.particles[..., :5][mask] = logit(self.particles[..., :5][mask], alpha=alpha)
            print('INFO: applying logit transform alpha={}'.format(alpha))
        else:
            pass
            # self.particles[..., :5] = expit(self.particles[..., :5][mask], alpha=alpha)
            # print('INFO: applying expit transform alpha={}'.format(alpha))

    def preprocess(self, methods: dict={}):
        method_items = list(methods.items())  
        for method_name, method_kwargs in method_items:
            method_kwargs['inverse'] = False
            method = getattr(self, method_name, None)
            if method and callable(method):
                method(**method_kwargs)
            else:
                print(f"Method {method_name} not found")

    def postprocess(self, methods: dict={}):
        method_items = list(methods.items())  
        method_items.reverse()
        for method_name, method_kwargs in method_items:
            method_kwargs['inverse'] = True
            method = getattr(self, method_name, None)
            if method and callable(method):
                method(**method_kwargs)
            else:
                print(f"Method {method_name} not found")

    #...ploting methods
    
    def particle_plot(self, 
                      feature, 
                      nth_particle=None, 
                      masked=True, 
                      bins=100, 
                      color='k',
                      xlog=False,
                      ylog=True, 
                      fill=True, 
                      ax=None,  
                      figsize=(3,3)): 
        
        idx = {'eta_rel':0, 'phi_rel':1, 'pt_rel':2, 'R':3}        
        if nth_particle is None:
            x = self.particles[..., idx[feature]] 
            mask = self.particles[..., -1] > 0
            x = x[mask].flatten() if masked else x.flatten() 
        else:
            x = self.nth_particle(n=nth_particle, feature=feature)
        fig, ax = plt.subplots(1, figsize=figsize) if ax is None else (None, ax)
        lw, alpha = (0.5, 0.2) if fill else (0.75, 1.0)
        sns.histplot(x=x, color=color, bins=bins, element="step", log_scale=(xlog, ylog), lw=lw, fill=fill, alpha=alpha, ax=ax) 
        plt.xlabel(r'{}'.format(feature))

    def jet_plot(self, 
                 feature, 
                 color='k', 
                 bins=100, 
                 xlog=False,
                 ylog=True,                
                 fill=True,
                 ax=None,
                 figsize=(3,3)):
        
        idx = {'pt_rel':0, 'eta_rel':1, 'phi_rel':2, 'm_rel':3, 'e_rel':4, 'multiplicity':5}
        x = self.jets[..., idx[feature]]
        fig, ax = plt.subplots(1, figsize=figsize) if ax is None else (None, ax)
        lw, alpha = (0.5, 0.2) if fill else (0.75, 1.0)
        sns.histplot(x=x, color=color, bins=bins, element="step", log_scale=(xlog, ylog), lw=lw, fill=fill, alpha=alpha, ax=ax) 
        plt.xlabel(r'{}'.format(feature))
       
        
    def image(self, event_num=None, figsize=(5,5), legend='brief'): 
        if event_num is not None: data = self.particles[event_num]
        else: data = self.particles
        
        mask = data[..., -1] > 0
        event = data[mask]
        event = torch.flip(event, dims=[0]) # invert pt ordering for improved visualization
        eta, phi, pt = event[...,0], event[...,1], event[...,2]
        fig, ax = plt.subplots(1, figsize=figsize)
        
        if event_num is not None:
            plt.title('event #{}'.format(event_num))
            sns.scatterplot(x=eta, y=phi, hue=pt, size=pt, palette='viridis', sizes=(10, 50), legend=legend) 
        else:
            plt.title('average events')
            sns.histplot(x=eta, y=phi, bins=(150,150), cmap='viridis')
            plt.xlim(-1,1)
            plt.ylim(-1,1)
        
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel(r'$\Delta\eta$')
        plt.ylabel(r'$\Delta\phi$')
        plt.show()
   
def logit(t, alpha=1e-6) -> torch.Tensor:
    x = alpha + (1 - 2 * alpha) * t
    return torch.log(x/(1-x))

# def expit(t,  alpha=1e-6) -> torch.Tensor:
#     exp = torch.exp(t)
#     x = exp / (1 + exp) 
#     return (x - mask * alpha) / (1 - 2*alpha)