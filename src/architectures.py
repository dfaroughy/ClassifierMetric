import torch
from torch import Tensor
import torch.nn as nn

#################################
'''
Possible classifier architectures:
    - Multi-Layer Perceptron
    - Deep Sets 
    TODO: fix Particle Net
    TODO: implement Transformer
'''
#################################

#...wrappers

class MLP(nn.Module):
    ''' Wrapper class for the MLP architecture'''
    def __init__(self, model_config):
        super(MLP, self).__init__()
        self.dim_features = model_config.dim_input
        self.device = model_config.device
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
        output = self.forward(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        return loss

    @torch.no_grad()
    def predict(self, batch): 
        data = batch['jet_features'].to(self.device)
        logits = self.forward(data)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs 

class DeepSets(nn.Module):
    ''' Wrapper class for the Deep Sets architecture'''
    def __init__(self, model_config):
        super(DeepSets, self).__init__()
        self.dim_features = model_config.dim_input
        self.device = model_config.device
        self.wrapper = _DeepSets(dim=model_config.dim_input, 
                                num_classes=model_config.dim_output,
                                dim_hidden=model_config.dim_hidden, 
                                num_layers_1=model_config.num_layers_1,
                                num_layers_2=model_config.num_layers_2,
                                device=model_config.device)
    def forward(self, x):
        return self.wrapper.forward(x)
    
    def loss(self, batch):
        data = batch['particle_features'].to(self.device)
        labels = batch['label'].to(self.device)
        output = self.forward(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        return loss

    @torch.no_grad()
    def predict(self, batch): 
        data = batch['particle_features'].to(self.device)
        logits = self.forward(data)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs 

#...architecture classes

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

    def forward(self, x):
        return self.net(x)
    

class _DeepSets(nn.Module):

    def __init__(self, 
                 dim, 
                 num_classes, 
                 dim_hidden=128, 
                 num_layers_1=2, 
                 num_layers_2=2, 
                 device='cpu'):

        super(_DeepSets, self).__init__()

        self.device = device
        self.dim = dim
        layers_1 = [nn.Linear(self.dim, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers_1 - 1)
        layers_2 = [nn.Linear(2 * dim_hidden, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers_2 - 1) + [nn.Linear(dim_hidden, num_classes), nn.LeakyReLU()]

        self.phi = nn.Sequential(*layers_1).to(device)
        self.rho = nn.Sequential(*layers_2).to(device)

    def forward(self, x): 
        h = self.phi(x)                                                      # shape: (N, m, hidden_dim)
        h = torch.cat([torch.sum(h, dim=1), torch.mean(h, dim=1)], dim=1)    # sum and mean pooling shape: (N, hidden_dim)
        h = self.rho(h)                                                      # shape: (N, output_dim)
        return h


# class ParticleNet(nn.Module):
#     ''' Wrapper class for the Particle Net architecture
#     '''
#     def __init__(self, config):
#         super(ParticleNet, self).__init__()
#         self.dim_features = config.dim_input
#         self.device = config.device
    
#         self.wrapper = _ParticleNet(num_hits, node_feat_size, num_classes, device=config.device)
#     def forward(self, x, mask):
#         points = x[..., :2] # (eta, phi)
#         features = x[..., 2 : self.dim_features] # (pt, pt_rel, R)
#         return self.wrapper.forward(points, features, mask)
    
#     def loss(self, batch):
#         batch = batch.to(self.device)
#         features, mask, labels = batch[..., :self.dim_features], batch[..., -2, None], batch[..., -1].long()[:,0]
#         output = self.forward(features, mask)
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(output, labels)
#         return loss


# class _ParticleEdgeBlock(nn.Module):

#     def __init__(self, 
#                 dim_input, 
#                 dim_output, 
#                 num_knn, 
#                 batch_norm=True, 
#                 activation=True, 
#                 device='cpu'):

#         super(_ParticleEdgeBlock, self).__init__()

#         self.num_knn = num_knn
#         self.batch_norm = batch_norm
#         self.activation = activation
#         self.num_layers = len(dim_output)
#         self.device = device

#         self.convs = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.convs.append(nn.Conv2d(2 * dim_input if i == 0 else dim_output[i - 1], dim_output[i], kernel_size=1, bias=False if self.batch_norm else True))
        
#         if batch_norm:
#             self.bns = nn.ModuleList()
#             for i in range(self.num_layers):
#                 self.bns.append(nn.BatchNorm2d(dim_output[i]))
        
#         if activation:
#             self.acts = nn.ModuleList()
#             for i in range(self.num_layers):
#                 self.acts.append(nn.ReLU())
        
#         if dim_input == dim_output[-1]:
#             self.sc = None
#         else:
#             self.sc = nn.Conv1d(dim_input, dim_output[-1], kernel_size=1, bias=False)
#             self.sc_bn = nn.BatchNorm1d(dim_output[-1])
        
#         if activation:
#             self.sc_act = nn.ReLU()

#     def knn(self, x):
#         xx = -2 * torch.matmul(x.transpose(2, 1), x)
#         x2 = torch.sum(x**2, dim=1, keepdim=True)
#         pairwise_distance = -x2 - xx - x2.transpose(2, 1)
#         idx = pairwise_distance.topk(k=self.num_knn + 1, dim=-1)[1][:, :, 1:]
#         return idx

#     def get_graph_feature(self, x, idx):
#         batch_size, dim, num_points = x.size()
#         idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
#         idx += idx_base
#         idx = idx.view(-1)
#         if self.device == 'cpu':
#             fts = x.transpose(0, 1).reshape(dim, -1)
#             fts = fts[:, idx].view(dim, batch_size, num_points, self.num_knn)
#             fts = fts.transpose(1, 0).contiguous()
#         else:  
#             fts = x.transpose(2, 1).reshape(-1, dim)
#             fts = fts[idx, :].view(batch_size, num_points, self.num_knn, dim)
#             fts = fts.permute(0, 3, 1, 2).contiguous()

#         x = x.view(batch_size, dim, num_points, 1).repeat(1, 1, 1, self.num_knn)
#         features = torch.cat((x, fts - x), dim=1)
#         return features

#     def forward(self, points, features):
#         indx = self.knn(points)
#         graph = self.get_graph_feature(features, indx)

#         for conv, batch_norm, activation in zip(self.convs, self.bns, self.acts):
#             g = conv(graph)
#             if batch_norm: g = batch_norm(g)
#             if activation: g = activation(g)

#         f = g.mean(dim=-1)

#         if self.sc:
#             h = self.sc(features)
#             h = self.sc_bn(h)
#         else:
#             h = features

#         return self.sc_act(h + f)


# class _ParticleNet(nn.Module):

#     def __init__(self, 
#                 dim_features, 
#                 num_classes, 
#                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
#                 fc_params=[(128, 0.1)], 
#                 use_fusion_block=False, 
#                 use_batch_norm=True, 
#                 use_counts=True, 
#                 device='cpu'):

#         super(_ParticleNet, self).__init__()

#         self.device = device
#         self.use_fusion_block = use_fusion_block
#         self.use_batch_norm = use_batch_norm
#         self.use_counts = use_counts

#         # batch normalization

#         if self.use_batch_norm:
#             self.batch_norm_layer = nn.BatchNorm1d(dim_features).to(device)

#         # edge convs

#         self.edge_convs = nn.ModuleList()
#         for idx, layer_param in enumerate(conv_params):
#             k, channels = layer_param
#             dim_input = dim_features if idx == 0 else conv_params[idx - 1][1][-1]
#             self.edge_convs.append(_ParticleEdgeBlock(k=k, dim_input=dim_input, dim_output=channels, device=device).to(device))

#         # fusion block

#         if self.use_fusion:
#             in_chn = sum(x[-1] for _, x in conv_params)
#             out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
#             self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), 
#                                               nn.BatchNorm1d(out_chn), 
#                                               nn.ReLU()).to(device)
#         # fully connected layers

#         fcs = []
#         for idx, layer_param in enumerate(fc_params):
#             channels, drop_rate = layer_param
#             if idx == 0:  in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
#             else: in_chn = fc_params[idx - 1][0]
#             fcs.append( nn.Sequential( nn.Linear(in_chn, channels),
#                                        nn.ReLU(),
#                                        nn.Dropout(drop_rate)))
#         fcs.append(nn.Linear(fc_params[-1][0], num_classes))
#         self.fc = nn.Sequential(*fcs).to(device)

#     def forward(self, points, features, mask):  
            
#         points *= mask
#         features *= mask
#         coord_shift = (mask == 0) * 1e9
        
#         if self.use_counts:
#             counts = mask.float().sum(dim=-1)
#             counts = torch.max(counts, torch.ones_like(counts))  # >=1

#         f = self.batch_norm_layer(features) * mask if self.use_batch_norm else features

#         outputs = []
#         for idx, conv in enumerate(self.edge_convs):
#             pts = (points if idx == 0 else f) + coord_shift
#             f = conv(pts, f) * mask
#             if self.use_fusion: outputs.append(f)

#         f = self.fusion_block(torch.cat(outputs, dim=1)) * mask if self.use_fusion else f
#         h = f.sum(dim=-1) / counts if self.use_counts else f.mean(dim=-1)
#         h = self.fc(h)
#         return h

#     def loss(self, batch): 
#         batch = batch.to(self.device)
#         coords = batch[..., 0:2].transpose(1, 2)
#         features = batch[..., 2:5].transpose(1, 2)
#         mask = batch[..., -1, None].transpose(1, 2)
#         labels = batch[..., -1].long()[:,0]
#         criterion = nn.CrossEntropyLoss()
#         output = self.forward(points=coords, features=features, mask=mask)
#         loss = criterion(output, labels)
#         return loss

#     @torch.no_grad()
#     def probability(self, x, batch_size=1024): 
#         num_batches = x.shape[0] // batch_size
#         batches = torch.chunk(x, num_batches, dim=0)
#         probs = []
#         for batch in batches:
#             batch  = batch.to(self.device)
#             coords = batch[..., 0:2].transpose(1, 2)
#             features = batch[..., 2:4].transpose(1, 2)
#             mask = batch[..., -1, None].transpose(1, 2)
#             logits = self.forward(points=coords, features=features, mask=mask)
#             batch_prob = torch.nn.functional.softmax(logits, dim=1)
#             probs.append(batch_prob)
#         return torch.cat(probs, dim=0).detach().cpu() 

