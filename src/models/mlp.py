import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class _MLP(nn.Module):

    def __init__(self, 
                dim, 
                dim_hidden,
                num_layers, 
                num_classes, 
                device='cpu'):

        super(_MLP, self).__init__()

        self.device = device
        self.layers = [nn.Linear(dim, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers - 1) + [nn.Linear(dim_hidden, num_classes), nn.LeakyReLU()]
        self.net = nn.Sequential(*self.layers).to(device)

    def forward(self, feautures):
        return self.net(feautures)
    

class MLP(nn.Module):
    ''' Wrapper class for the MLP architecture'''
    def __init__(self, model_config):
        super(MLP, self).__init__()
        self.dim_features = model_config.dim_input
        self.device = model_config.device
        self.criterion = nn.CrossEntropyLoss()
        self.wrapper = _MLP(dim=model_config.dim_input, 
                            num_classes=model_config.dim_output,
                            dim_hidden=model_config.dim_hidden, 
                            num_layers=model_config.num_layers,
                            device=model_config.device)
    def forward(self, x):
        return self.wrapper.forward(x)
    
    def loss(self, batch):
        data = batch['jet_features'].to(self.device)
        labels = batch['label'].to(self.device)
        output = self.forward(x=data)
        loss = self.criterion(output, labels)
        return loss

    @torch.no_grad()
    def predict(self, batch): 
        data = batch['jet_features'].to(self.device)
        logits = self.forward(data)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()  