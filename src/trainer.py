import torch
import torch.nn as nn
import numpy as np
# from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm.auto import tqdm
from src.plots import plot_loss
from src.datamodule.dataloaders import JetNetDataLoader

class ModelClassifierTest:

    def __init__(self, 
                 classifier, 
                 dataloader: JetNetDataLoader=None,
                 epochs: int=100, 
                 lr: float=0.001, 
                 early_stopping : int=10,
                 warmup_epochs: int=20,
                 workdir: str='./',
                 seed=12345):
    
        super(ModelClassifierTest, self).__init__()

        self.model = classifier
        self.train_loader = dataloader.train_loader
        self.valid_loader = dataloader.valid_loader
        self.test_loader = dataloader.test_loader
        self.workdir = workdir
        self.lr = lr
        self.seed = seed
        self.early_stopping = early_stopping 
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs

    def train(self):
        train = Train_Step(loss_fn=self.model.loss)
        valid = Validation_Step(loss_fn=self.model.loss, warmup_epochs=self.warmup_epochs)
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

            if epoch % 4 == 0: plot_loss(train, valid, workdir=self.workdir)
        plot_loss(train, valid, workdir=self.workdir)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.model.device)))


    @torch.no_grad()
    def test(self, class_labels: dict=None):
        self.predictions = {}
        self.log_posterior = {}
        temp = []

        for batch in tqdm(self.test_loader, desc="testing"):
            prob = self.model.predict(batch)
            res = torch.cat([prob, batch['label'].unsqueeze(-1)], dim=-1)
            temp.append(res)

        self.predictions['datasets'] = torch.cat(temp, dim=0) 
        labels = self.predictions['datasets'][:, -1] 

        for _, label in class_labels.items():
            self.predictions[label] = self.predictions['datasets'][labels == label][:, :-1]
            if label != -1: self.log_posterior[label] = torch.log(self.predictions[label]).mean(dim=0)


############################


class Train_Step(nn.Module):

    def __init__(self, loss_fn):
        super(Train_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.print_epoch = 10
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

        if self.epoch % self.print_epoch  == 0:
            print("\t Training loss: {}".format(self.loss))

        self.losses.append(self.loss) 


class Validation_Step(nn.Module):

    def __init__(self, loss_fn, warmup_epochs=10):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.terminate_loop = False
        self.data_size = 0
        self.print_epoch = 5
        self.warmup_epochs = warmup_epochs
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
                
                torch.save(save_best.state_dict(), workdir + '/best_model.pth')    
            else: self.patience += 1 if self.epoch > self.warmup_epochs else 0
            if self.patience >= early_stopping: self.terminate_loop = True
        else:
            torch.save(save_best.state_dict(), workdir + '/best_model.pth')
        if self.epoch % self.print_epoch == 1:
            print("\t Test loss: {}  (min loss: {})".format(self.loss, self.loss_min))
        return self.terminate_loop
