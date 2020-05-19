# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:39:06 2020

@author: SFC202004009
"""


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from utils import read_ZINC_smiles, smiles_to_onehot, smiles_to_onehot2
from utils2 import read_smiles, get_logP, smiles_to_number
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys
import time
import argparse

class DataSet(Dataset):
    def __init__(self, smi_list, logP_list, len_list):
        self.smi_list = smi_list
        self.logP_list = logP_list
        self.len_list = len_list
        
    def __len__(self):
        return len(self.logP_list)
    
    def __getitem__(self, index):
        return self.smi_list[index], self.logP_list[index], self.len_list[index]
    
    
def make_partition():
    # smi_list, logP_total, tpsa_total = read_ZINC_smiles(50000)
    smi_list = read_smiles('ZINC.smiles', 50000)
    logP_total = get_logP(smi_list)
    smi_total = smiles_to_number(smi_list, 120)
    len_total = [len(x) for x in smi_list]
    
    logP_total = np.array(logP_total)
    smi_total = np.array(smi_total)
    len_total = np.array(len_total)
    print(len_total.shape)
    print(smi_total.shape)
    print(logP_total.shape)
    # smi_list = []
    # with open("smi_list.txt", "r") as f:
    #   for line in f:
    #     smi_list.append(str(line.strip()))
    # logP_total = np.load('logP_list.npy')
    # smi_total = smiles_to_onehot(smi_list)
    
    # smi_total, len_total = smiles_to_onehot2(smi_list)
    # print(len_total.shape)
    # print(smi_total.shape)
    # print(logP_total.shape)
    
    num_train = 30000
    num_validation = 10000
    num_test = 10000
    
    smi_train = smi_total[0:num_train]
    logP_train = logP_total[0:num_train]
    len_train = len_total[0:num_train]
    smi_validation = smi_total[num_train:(num_train+num_validation)]
    logP_validation = logP_total[num_train:(num_train+num_validation)]
    len_validation = len_total[num_train:(num_train+num_validation)]
    smi_test = smi_total[(num_train+num_validation):]
    logP_test = logP_total[(num_train+num_validation):]
    len_test = len_total[(num_train+num_validation):]

    
    train_dataset = DataSet(smi_train, logP_train, len_train)
    val_dataset = DataSet(smi_validation, logP_validation, len_validation)
    test_dataset = DataSet(smi_test, logP_test, len_test)
    
    partition = {'train': train_dataset,
                 'val': val_dataset,
                 'test': test_dataset}
    
    return partition

class Rnn(nn.Module):
    def __init__(self, hidden_dim):
        super(Rnn, self).__init__()
        self.rnn = nn.GRU(31, hidden_dim, batch_first=True)
        
        self.linear1 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.embed = nn.Embedding(31, 31, padding_idx=0)
        #self.rnn = nn.GRU(31, 256,num_layers=1, bias=True,batch_first=True,bidirectional=True)
        
    def forward(self, x, lengths):
        x = x.long()
        x = self.embed(x)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.tolist(), batch_first=True)
        x, h = self.rnn(packed_input)
        # x = x[0]
        x = h
        x = x.squeeze()

        x = self.relu( self.linear1(x) ) 
        x = self.tanh( self.linear2(x) ) 
        
        x = self.output(x)
        return x
    
    def xavier_init(self):        
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.output.weight)
        self.output.bias.data.fill_(0.01)

def make_one_hot(nump, num):
    one = np.identity(31, dtype = np.float)
    nump_one_hot = np.zeros((nump.shape[0], nump.shape[1], num), dtype = np.float)
    for i_n, i in enumerate(nump):
        for j_n, j in enumerate(i):
            nump_one_hot[i_n, j_n] = one[j]
    return nump_one_hot

def train(net,partition, optimizer, criterion):
    dataloader = DataLoader(partition['train'], batch_size = 128, shuffle = True, num_workers = 0)
    
    net.train()
    optimizer.zero_grad()
    
    total = 0
    train_loss = 0.0
    for data in dataloader:
        x, labels, lengths = data

        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]
        labels = labels[sorted_idx]
            
        # x= make_one_hot(x, 31)
        # x = torch.from_numpy(x)
        x = x.float()
        x = x.cuda()
        labels = labels.float()
        labels = labels.cuda()
        
        outputs = net(x, lengths)
        
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total += labels.size(0)
        
    train_loss /= len(dataloader)
    
    return net, train_loss

def validate(net, partition, criterion):
    dataloader = DataLoader(partition['val'], batch_size = 10000, shuffle = True, num_workers = 0)
    net.eval()
    
    with torch.no_grad():
        total = 0
        val_loss = 0.0
        for data in dataloader:
            x, labels, lengths = data
            
            lengths, sorted_idx = lengths.sort(0, descending=True)
            x = x[sorted_idx]
            labels = labels[sorted_idx]
            
            # x= make_one_hot(x, 31)
            # x = torch.from_numpy(x)
            x = x.float()
            x = x.cuda()
            labels = labels.float()
            labels = labels.cuda()
            
            outputs = net(x, lengths)
            
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    
    return val_loss

def test(net, partition, criterion):
    dataloader = DataLoader(partition['test'], batch_size = 10000, shuffle = True, num_workers = 0)
    net.eval()
    
    with torch.no_grad():
        total = 0
        test_loss = 0.0
        for data in dataloader:
            x, labels, lengths = data
            lengths, sorted_idx = lengths.sort(0, descending=True)
            x = x[sorted_idx]
            labels = labels[sorted_idx]
            # x= make_one_hot(x, 31)
            # x = torch.from_numpy(x)
            x = x.float()
            x = x.cuda()
            labels = labels.float()
            labels = labels.cuda()
            
            outputs = net(x, lengths)
            
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            test_loss  += loss.item()
    test_loss  /= len(dataloader)
    
    labels = labels.cpu()
    outputs = outputs.cpu()
    
    plt.figure()
    plt.scatter(labels, outputs, s=3)
    plt.xlabel('logP - Truth', fontsize=15)
    plt.ylabel('logP - Prediction', fontsize=15)
    x = np.arange(-4,6)
    plt.plot(x,x,c='black')
    plt.tight_layout()
    plt.axis('equal')
    plt.show()
    
    return test_loss 

def experiment(args):
    net = Rnn(args.hidden_dim)
    partition = make_partition()
    criterion = nn.MSELoss()
    
    if(args.optim == 'sgd'):
        optimizer = optim.SGD(net.parameters(), lr = args.lr, weight_decay = 1e-4)
    elif(args.optim == 'adam'):
        optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = 1e-4)
    
    net.cuda()
    
    epoch_size = args.epoch_size
    for i in range(epoch_size):
        for g in optimizer.param_groups:
            g['lr'] = args.lr * 0.99**i
            
        tic = time.time()
        net, train_loss = train(net, partition, optimizer, criterion)
        val_loss = validate(net, partition, criterion)
        tok = time.time()
        print("epoch: {} test loss: {:2.3f}, val_loss: {:2.3f} took: {:2.2f}".format(i, train_loss, val_loss, tok-tic))
        
    test_loss = test(net, partition, criterion)
    print("test loss: {}".format(test_loss))
    
# seed = 690220955545900
# seed2 = 3707523286557544
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed2)

print('torch seed: ', torch.initial_seed())
print('torch cuda seed: ', torch.cuda.initial_seed())

parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.lr = 0.005
args.epoch_size = 50

args.hidden_dim = 256

args.optim = 'sgd' #sgd, adam

experiment(args)
