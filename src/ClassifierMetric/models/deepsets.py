import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#...architecture classes

class _DeepSets(nn.Module):

    def __init__(self, 
                 dim=3, 
                 num_classes=2, 
                 dim_hidden=128, 
                 num_layers_1=2, 
                 num_layers_2=2, 
                 device='cpu'):

        super(_DeepSets, self).__init__()

        self.device = device
        self.dim = dim
        layers_1 = [nn.Linear(dim, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers_1 - 1)
        layers_2 = [nn.Linear(2 * dim_hidden, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers_2 - 2) + [nn.Linear(dim_hidden, num_classes), nn.LeakyReLU()]
        self.phi = nn.Sequential(*layers_1).to(device)
        self.rho = nn.Sequential(*layers_2).to(device)

    def forward(self, features, mask=None):
        mask = mask.unsqueeze(-1) if mask is not None else torch.ones_like(features[..., 0]).unsqueeze(-1)
        h = self.phi(features)                               # shape: (batch_size, num_consts, dim_hidden)
        h_sum = (h * mask).sum(1, keepdim=False)
        h_mean = h_sum / mask.sum(1, keepdim=False)
        h_meansum_pool = torch.cat([h_mean, h_sum], dim=1)   # shape: (batch_size, 2*dim_hidden)
        f = self.rho(h_meansum_pool)                         # shape: (batch_size, num_classes)
        return f


class DeepSets(nn.Module):
    ''' Wrapper class for the Deep Sets architecture'''
    def __init__(self, model_config):
        super(DeepSets, self).__init__()

        self.dim_features = model_config.dim_input
        self.device = model_config.device

        self.deepset = _DeepSets(dim=model_config.dim_input, 
                                num_classes=model_config.dim_output,
                                dim_hidden=model_config.dim_hidden, 
                                num_layers_1=model_config.num_layers_1,
                                num_layers_2=model_config.num_layers_2,
                                device=model_config.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features, mask):
        return self.deepset.forward(features, mask)
    
    def loss(self, batch):
        features = batch['particle_features'].to(self.device)
        labels = batch['label'].to(self.device)
        mask = batch['mask'].to(self.device)
        output = self.forward(features, mask)
        loss =  self.criterion(output, labels)
        return loss

    @torch.no_grad()
    def predict(self, batch): 
        features = batch['particle_features'].to(self.device)
        mask = batch['mask'].to(self.device)
        logits = self.forward(features, mask)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()  