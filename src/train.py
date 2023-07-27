import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm
from src.plots import plot_loss
from src.datasets import JetNetDataset

class ModelClassifierTest:

    def __init__(self, 
                 classifier, 
                 datasets: JetNetDataset=None,
                 truth_label: int=None,
                 split_fractions: tuple=None,
                 epochs: int=100, 
                 lr: float=0.001, 
                 early_stopping : int=10,
                 workdir: str='./',
                 seed=12345):
    
        super(ModelClassifierTest, self).__init__()

        self.datasets = datasets        
        self.split_fractions = split_fractions
        self.truth_label = truth_label
        self.model = classifier
        self.workdir = workdir
        self.lr = lr
        self.seed = seed
        self.early_stopping = early_stopping 
        self.epochs = epochs


    def train_val_test_split(self, dataset, train_frac, valid_frac):

        assert sum(self.split_fractions) - 1.0 < 1e-3, "Split fractions do not sum to 1!"
        total_size = len(dataset)
        train_size = int(total_size * train_frac)
        valid_size = int(total_size * valid_frac)
        
        #...define splitting indices
        
        idx = torch.arange(total_size)        
        idx_train = idx[:train_size]
        idx_valid = idx[train_size : train_size + valid_size]
        idx_test = idx[train_size + valid_size :]
        
        #...Create Subset for each split
        
        train_set = Subset(dataset, idx_train)
        valid_set = Subset(dataset, idx_valid)
        test_set = Subset(dataset, idx_test)

        return train_set, valid_set, test_set


    def DataLoaders(self, batch_size):

        #...split datasets into truth data / models data

        labels = [item['label'] for item in self.datasets]
        truth_label = np.max(labels) if self.truth_label is None else self.truth_label
        idx_truth = [i for i, label in enumerate(labels) if label == truth_label]
        idx_models = [i for i, label in enumerate(labels) if label != truth_label]
        samples_truth = Subset(self.datasets, idx_truth)
        samples_models = Subset(self.datasets, idx_models)

        #...get training / validation / test samples   

        print("INFO: train/val/test split ratios of {} / {} / {}".format(self.split_fractions[0], self.split_fractions[1], self.split_fractions[2]))
        train_models, valid_models, test_models = self.train_val_test_split(dataset=samples_models, 
                                                                            train_frac=self.split_fractions[0], 
                                                                            valid_frac=self.split_fractions[1])
        _train_truth, _valid_truth, test_truth  = self.train_val_test_split(dataset=samples_truth, 
                                                                            train_frac=self.split_fractions[0], 
                                                                            valid_frac=self.split_fractions[1])
        test = ConcatDataset([test_truth, test_models])

        #...create dataloaders

        self.train_loader = DataLoader(dataset=train_models, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_models,  batch_size=batch_size, shuffle=False)
        self.truth_loader = DataLoader(dataset=test,  batch_size=batch_size, shuffle=True)


    def train(self):

        train = Train_Step(loss_fn=self.model.loss)
        valid = Validation_Step(loss_fn=self.model.loss)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))

        for epoch in tqdm(range(self.epochs), desc="epochs"):

            train.update(data=self.train_loader, optimizer=optimizer)       
            valid.update(data=self.valid_loader)
            scheduler.step() 

            if valid.stop(save_best=self.model,
                          early_stopping =self.early_stopping, 
                          workdir=self.workdir): 
                print("INFO: early stopping triggered! Reached maximum patience at {} epochs".format(epoch))
                break

            if epoch % 5 == 1: plot_loss(train, valid, workdir=self.workdir)
        plot_loss(train, valid, workdir=self.workdir)


    def test(self):
        for batch in self.truth_loader:
            prob = self.model.predict(batch)
            print(prob.shape, batch['label'].shape)
            print(torch.cat([prob, batch['label']], dim=-1))

class Train_Step(nn.Module):

    def __init__(self, loss_fn):
        super(Train_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.print_epoch = 5
        self.losses = []

    def update(self, data: torch.Tensor, optimizer):
        self.loss = 0
        self.epoch += 1

        for batch in data:
            optimizer.zero_grad()
            loss_current = self.loss_fn(batch)
            loss_current.backward()
            optimizer.step()  
            self.loss += loss_current.detach().cpu().numpy()

        self.loss = self.loss / len(data)

        if self.epoch % self.print_epoch  == 1:
            print("\t Training loss: {}".format(self.loss))
        self.losses.append(self.loss) 


class Validation_Step(nn.Module):

    def __init__(self, loss_fn, model_name='best'):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.name = model_name
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.terminate_loop = False
        self.data_size = 0
        self.print_epoch = 5
        self.losses = []

    @torch.no_grad()
    def update(self, data: torch.Tensor):
        self.loss = 0
        self.epoch += 1

        for batch in data:
            loss_current = self.loss_fn(batch)
            self.loss += loss_current.detach().cpu().numpy()

        self.loss = self.loss / len(data)
        self.losses.append(self.loss) 

    @torch.no_grad()
    def stop(self, save_best, early_stopping, workdir):
        if early_stopping is not None:
            if self.loss < self.loss_min:
                self.loss_min = self.loss
                self.patience = 0
                torch.save(save_best.state_dict(), workdir + '/{}_model.pth'.format(self.name))    
            else: self.patience += 1
            if self.patience >= early_stopping: self.terminate_loop = True
        else:
            torch.save(save_best.state_dict(), workdir + '/{}_model.pth'.format(self.name))

        if self.epoch % self.print_epoch == 1:
            print("\t Test loss: {}  (min loss: {})".format(self.loss, self.loss_min))

        return self.terminate_loop


# class MultiClassifierTest:

#     def __init__(self, 
#                  classifier, 
#                  samples: tuple, 
#                  epochs: int=100, 
#                  lr: float=0.001, 
#                  early_stopping : int=10,
#                  workdir: str='./',
#                  seed=12345):
#         super(MultiClassifierTest, self).__init__()
        
#         self.model = classifier
#         self.workdir = workdir
#         self.lr = lr
#         self.seed = seed
#         self.early_stopping = early_stopping 
#         self.epochs = epochs
#         self.data = []   

#         for i, sample in enumerate(samples):
#             label =  torch.full((sample.shape[0], sample.shape[1], 1), i)
#             sample = torch.cat([sample, label.long()], dim=-1)
#             self.data.append(sample)
        
#         self.data = torch.cat(self.data, dim=0)
        
#     def DataLoader(self, test_size, batch_size):
#         train, test  = train_test_split(self.data, test_size=test_size, random_state=self.seed)
#         self.train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=batch_size, shuffle=True)
#         self.test_sample = DataLoader(dataset=torch.Tensor(test),  batch_size=batch_size, shuffle=False)
    
#     def train(self, model_name='best'):
        
#         train = Train_Step(loss_fn=self.model.loss)
#         test = Test_Step(loss_fn=self.model.loss, model_name=model_name)
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

#         print("INFO: start training") 
#         print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))

#         for epoch in tqdm(range(self.epochs), desc="epochs"):

#             train.update(data=self.train_sample, optimizer=optimizer)       
#             test.eval(data=self.test_sample)
#             scheduler.step() 

#             if test.improved(save_best=self.model,
#                              early_stopping =self.early_stopping , 
#                              workdir=self.workdir): 
#                 print("INFO: early stopping triggered! Reached maximum patience at {} epochs".format(epoch))
#                 break
#             if epoch % 5 == 1:
#                 plot_loss(train, test, workdir=self.workdir, overwrite=True)
#         plot_loss(train, test, workdir=self.workdir, overwrite=True)

# class Train_Step(nn.Module):

#     def __init__(self, loss_fn):
#         super(Train_Step, self).__init__()
#         self.loss_fn = loss_fn
#         self.loss = 0
#         self.epoch = 0
#         self.print_epoch = 5
#         self.losses = []

#     def update(self, data: Tensor, optimizer):
#         self.loss = 0
#         self.epoch += 1

#         for batch in data:
#             optimizer.zero_grad()
#             loss_current = self.loss_fn(batch)
#             loss_current.backward()
#             optimizer.step()  
#             self.loss += loss_current.detach().cpu().numpy()

#         self.loss = self.loss / len(data)

#         if self.epoch % self.print_epoch  == 1:
#             print("\t Training loss: {}".format(self.loss))
#         self.losses.append(self.loss) 


# class Test_Step(nn.Module):

#     def __init__(self, loss_fn, model_name='best'):
#         super(Test_Step, self).__init__()
#         self.loss_fn = loss_fn
#         self.name = model_name
#         self.loss = 0
#         self.epoch = 0
#         self.patience = 0
#         self.loss_min = np.inf
#         self.terminate_loop = False
#         self.data_size = 0
#         self.print_epoch = 5
#         self.losses = []

#     @torch.no_grad()
#     def eval(self, data: Tensor):
#         self.loss = 0
#         self.epoch += 1

#         for batch in data:
#             loss_current = self.loss_fn(batch)
#             self.loss += loss_current.detach().cpu().numpy()

#         self.loss = self.loss / len(data)
#         self.losses.append(self.loss) 

#     @torch.no_grad()
#     def improved(self, save_best, early_stopping, workdir):

#         if early_stopping is not None:
#             if self.loss < self.loss_min:
#                 self.loss_min = self.loss
#                 self.patience = 0
#                 torch.save(save_best.state_dict(), workdir + '/{}_model.pth'.format(self.name))    
#             else: self.patience += 1
#             if self.patience >= early_stopping: self.terminate_loop = True
#         else:
#             torch.save(save_best.state_dict(), workdir + '/{}_model.pth'.format(self.name))

#         if self.epoch % self.print_epoch == 1:
#             print("\t Test loss: {}  (min loss: {})".format(self.loss, self.loss_min))

#         return self.terminate_loop

