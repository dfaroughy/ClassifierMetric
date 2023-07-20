import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.plots import plot_loss
from datasets import JetNetDataSets


class MultiClassifierTest:

    def __init__(self, 
                 classifier, 
                 train_samples: JetNetDataSets = None,
                 test_samples: JetNetDataSets = None,
                 epochs: int=100, 
                 lr: float=0.001, 
                 early_stopping : int=10,
                 workdir: str='./',
                 seed=12345):
    
        super(MultiClassifierTest, self).__init__()
        
        self.model = classifier
        self.workdir = workdir
        self.lr = lr
        self.seed = seed
        self.early_stopping = early_stopping 
        self.epochs = epochs
        self.train_samples = train_samples   
        self.test_samples = test_samples   

    # def DataLoader(self, test_size, batch_size):
    #     train, test  = train_test_split(self.data, test_size=test_size, random_state=self.seed)
    #     self.train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=batch_size, shuffle=True)
    #     self.test_sample = DataLoader(dataset=torch.Tensor(test),  batch_size=batch_size, shuffle=False)
    
    
    def train(self):
        
        train = Train_Step(loss_fn=self.model.loss)
        valid = Validation_Step(loss_fn=self.model.loss)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        print("INFO: start training") 
        print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))

        for epoch in tqdm(range(self.epochs), desc="epochs"):

            train.update(data=self.train_sample, optimizer=optimizer)       
            valid.update(data=self.test_sample)
            scheduler.step() 

            if valid.stop(save_best=self.model,
                                early_stopping =self.early_stopping, 
                                workdir=self.workdir): 
                print("INFO: early stopping triggered! Reached maximum patience at {} epochs".format(epoch))
                break
            if epoch % 5 == 1:
                plot_loss(train, valid, workdir=self.workdir, overwrite=True)
        plot_loss(train, valid, workdir=self.workdir, overwrite=True)

    def test(self):
        pass

class Train_Step(nn.Module):

    def __init__(self, loss_fn):
        super(Train_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.print_epoch = 5
        self.losses = []

    def update(self, data: Tensor, optimizer):
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
    def update(self, data: Tensor):
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

