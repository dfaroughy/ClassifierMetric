import torch

class PreprocessData:

    ''' Module for preprocessing individual jets. 
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
                 stats: tuple=None,
                 compute_jet_features: bool=False):
        
        self.particles_features = data
        self.dim = self.particles_features.shape[-1] - 1
        if stats is not None:
            self.mean, self.std, self.min, self.max = stats 
        self.mask = self.particles_features[:, -1].unsqueeze(-1)
        self.particles_unmask = self.particles_features[:, :self.dim]
        if compute_jet_features:
            self.jet_features = self.get_jet_features()
    
    def get_jet_features(self):
        mask = self.mask.squeeze(-1)
        eta, phi, pt = self.particles_features[:, 0], self.particles_features[:, 1], self.particles_features[:, 2]
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

    def preprocess(self, methods: list=None):
        print('INFO: preprocessing data')
        for method in methods:
            method = getattr(self, method, None)
            if method and callable(method):
                method()
            else:
                print(f"Method {method} not found")
        
    def center_jets(self):
        N = self.particles_features.shape[0]
        jet_coords = self.jet_features[1:3] # jet (eta, phi)
        jet_coords = jet_coords.repeat(N, 1) * self.mask
        zeros = torch.zeros((N, self.dim - 2))
        jet_coords = torch.cat((jet_coords, zeros), dim=1)
        self.particles_unmask -= jet_coords 
        self.particles_unmask -= jet_coords 
        self.particles_features = torch.cat((self.particles_unmask, self.mask), dim=-1)

    def standardize(self,  sigma: float=1.0):
        self.particles_unmask = (self.particles_unmask * self.mean) * (1e-8 + sigma / self.std )
        self.particles_unmask = self.particles_unmask * self.mask
        self.particles_features = torch.cat((self.particles_unmask, self.mask), dim=-1)

    def normalize(self):
        self.particles_unmask = (self.particles_unmask - self.min) / ( self.max - self.min )
        self.particles_unmask = self.particles_unmask * self.mask
        self.particles_features = torch.cat((self.particles_unmask, self.mask), dim=-1)
    
    def logit_transform(self):
        self.particles_unmask = self.logit(self.particles_unmask)
        self.particles_unmask = self.particles_unmask * self.mask
        self.particles_features = torch.cat((self.particles_unmask, self.mask), dim=-1)

    def logit(t: torch.Tensor=None, alpha=1e-6):
        x = t * (1 - 2 * alpha) + alpha
        return torch.log(x/(1-x))