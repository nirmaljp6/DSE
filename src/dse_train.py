#This file trains the neural network in the deep state estimation (DSE)
#method of Nair and Goza, 2020.

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

#User inputs--------------------------------------
k = 25; #Number of POD modes
s = 5; #Number of sensors
#-------------------------------------------------

class sensorgcdataset(Dataset):
    
    def __init__(self, in_file, out_file, stats, transform=None):
        self.sensor_frame = pd.read_csv(in_file, header=None)
        self.gc_frame = pd.read_csv(out_file, header=None)
        self.stats = stats
        self.transform = transform
        
    def __len__(self):
        return len(self.sensor_frame)
    
    def __getitem__(self, idx):
        sensor = self.sensor_frame.iloc[idx,:].as_matrix()
        gc = self.gc_frame.iloc[idx,:].as_matrix()
        sensor = (sensor-self.stats[0][0,:])/self.stats[0][1,:]  
        gc = (gc-self.stats[1][0,:])/self.stats[1][1,:]   
        sensor = torch.from_numpy(sensor).float()
        gc = torch.from_numpy(gc).float()
        return sensor, gc
    
class network(nn.Module):
    
    def __init__(self, k, s):
        super(network, self).__init__()
        self.layer1 = nn.Linear(in_features=s, out_features=500, bias=True)
        self.layer2 = nn.Linear(in_features=500, out_features=500, bias=True)
        self.layer3 = nn.Linear(in_features=500, out_features=500, bias=True)
        self.output = nn.Linear(in_features=500, out_features=k, bias=True)
        self.dropout = nn.Dropout(p=0.2)   #dropout after relu
        self.bn = nn.BatchNorm1d(num_features=500)   #bn before relu
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

#------------------------------------------------------   
        
if __name__=="__main__":

  #Standardization---------------------
  stats_in = pd.read_csv('../preprocessed_data/stats_in.csv', header=None).as_matrix()
  stats_out = pd.read_csv('../preprocessed_data/stats_out.csv', header=None).as_matrix()
  stats = [stats_in, stats_out]
  
  #Reading data
  train = sensorgcdataset('../preprocessed_data/train_in.csv','../preprocessed_data/train_out.csv',stats)
  valid = sensorgcdataset('../preprocessed_data/valid_in.csv','../preprocessed_data/valid_out.csv',stats)
  
  bs = 80 #batch size
  train_data_gen = torch.utils.data.DataLoader(train,shuffle=True,batch_size=bs,num_workers=1)
  valid_data_gen = torch.utils.data.DataLoader(valid,batch_size=bs,num_workers=1)
  dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
  dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
  
  #Setting up the neural network
  model = network(k,s)
  
  # Loss and Optimizer
  criterion = nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1) 
          
  phi=pd.read_csv('../preprocessed_data/basis.csv', header=None).as_matrix()
  sensor_loc=pd.read_csv('../preprocessed_data/sensor_loc.csv', header=None).as_matrix().astype(int)
  
  num_epochs=5000
  prev_loss = 100   #large number
  stop=0
  for epochs in range(num_epochs):
      
      for phase in ['train','valid']:
          running_loss=0    
  
          if phase == 'train':
            exp_lr_scheduler.step()     
            model.train()
          else:
            model.eval()
          
          for i, (sensor,gc) in enumerate(dataloaders[phase]):
              
              sensor = Variable(sensor)
              gc = Variable(gc)
                          
              #clear gradients
              optimizer.zero_grad()
              #forward
              gc_out = model(sensor)
              #calculate loss
              loss = criterion(gc_out, gc)
              
              if phase=='train':    
                  #backward
                  loss.backward()
                  #update weights
                  optimizer.step()
              
              running_loss += gc_out.shape[0]*loss.data
          
          if phase == 'train':                    
              train_epoch_loss = running_loss/dataset_sizes[phase]
          elif phase == 'valid':
              valid_epoch_loss = running_loss/dataset_sizes[phase]
      stop = stop+1
      if (valid_epoch_loss < prev_loss):
        model_wts = model.state_dict()
        prev_loss = valid_epoch_loss
        print('({}) Training Loss: {:.8f} Valid Loss: {:.8f} *'.format(epochs, train_epoch_loss, valid_epoch_loss))
        stop = 0
      else:
        print('({}) Training Loss: {:.8f} Valid Loss: {:.8f} '.format(epochs, train_epoch_loss, valid_epoch_loss))
        
      if stop==100: 
        print('Early stopping criteria fulfilled')
        break
  
  torch.save(model_wts,'../nn_weights/nn_weights')
  
