#This file tests the deep state estimation (DSE)
#method of Nair and Goza, 2020.

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as pt

from dse_train import sensorgcdataset
from dse_train import network

#User input------------------------------------------
#Select case
aoa = 70; #choose 25 or 70 deg
k = 25; #Number of POD modes
s = 5; #Number of sensors
load_saved_weights = 1; #choose 0: newly trained weights or 1: pre-trained weights
if load_saved_weights == 1:
  pretrained_weights = '../nn_weights/weights_70_71_snaps400_forceL2_k25_s5'; #set the filename from nn_weights directory
#-------------------------------------------------

if aoa==25:
    nx=499; ny=399;
elif aoa==70:
    nx=399; ny=599;

#Standardization---------------------
stats_in = pd.read_csv('../preprocessed_data/stats_in.csv', header=None).as_matrix()
stats_out = pd.read_csv('../preprocessed_data/stats_out.csv', header=None).as_matrix()
stats = [stats_in, stats_out]

#Setting up the neural network
model = network(k,s)

phi=pd.read_csv('../preprocessed_data/basis.csv', header=None).as_matrix()
sensor_loc=pd.read_csv('../preprocessed_data/sensor_loc.csv', header=None).as_matrix().astype(int)

#Load saved neural network weights
if load_saved_weights == 0:
  model.load_state_dict(torch.load('../nn_weights/nn_weights'))
else:
  model.load_state_dict(torch.load(pretrained_weights))
model.eval()

#Load snapshots of sensor data and gc for testing
test = sensorgcdataset('../preprocessed_data/test_in.csv','../preprocessed_data/test_out.csv',stats)

#predicting generalized coordinates from sensor data
err=[]
gc_pred=np.zeros([len(test), model.output.out_features])
for i, (sensor,gc) in enumerate(test):
    gc_out = model(sensor[None,:]).data.numpy()
    gc_pred[i,:] = gc_out
    err.append( np.linalg.norm(gc.numpy()-gc_pred[i,:])**2/gc_pred[i,:].size)#/np.linalg.norm(gc) )

acc_avg = np.mean(err)
gc_pred = gc_pred*stats[1][1,:] + stats[1][0,:] 

#Loading full-state data
Ttrue = pd.read_csv('../preprocessed_data/state.csv', header=None).as_matrix()
Tmean = pd.read_csv('../preprocessed_data/state_mean.csv', header=None).as_matrix()
Tpred = Tmean + gc_pred.dot(phi.T)
ns=nx*ny; i1=0; iend = ns; 
Terr=[]
for i in range(len(test)):
    Terr.append (np.linalg.norm(Ttrue[i,i1:iend]-Tpred[i,i1:iend])/np.linalg.norm(Ttrue[i,i1:iend])*100 ) 

Terr_avg = np.mean(Terr)
Terr_max=max(Terr)
print("Solution error =",Terr_avg)
print("Max error =",Terr_max)
i=Terr.index(max(Terr))

#Plotting
sensor_ind = pd.read_csv('../preprocessed_data/sensor_index.csv', header=None).as_matrix()
pt.figure()
pt.pcolormesh(Ttrue[i,:ns].reshape((ny,nx),order='F'),cmap=pt.cm.jet, vmin=-0.0003, vmax=0.0003)
pt.scatter(sensor_ind[0,:], sensor_ind[1,:], color='black', s=3)

pt.figure()
pt.pcolormesh(Tpred[i,:ns].reshape((ny,nx),order='F'),cmap=pt.cm.jet, vmin=-0.0003, vmax=0.0003)
pt.scatter(sensor_ind[0,:], sensor_ind[1,:], color='black', s=3)



