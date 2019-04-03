#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:56:33 2019

@author: lumina
"""
#importing the dependcies
from data_loader import imdb_dataset
from torch.utils.data import DataLoader
from data_preprocessing import count_of_csv
from RNN_Module import RNNMod
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import pandas as pd
 
#Insitantiate the dataloading class    
train_loader = imdb_dataset('train_data.csv')

# load the dataset in batches using pytorch Dataloader in built class
train_dataloader = DataLoader(dataset = train_loader, batch_size = 1024, shuffle = True, num_workers = 4)

#def transpose_batch(X, y):
#    return X.transpose(0,1), y

#Just a Check  whether the dataloader is working
#it = iter(train_dataloader)
#xs,ys = next(it)

#print('Batch Size', len(xs))
#print(type(xs))
#print(xs)
#Taking the vocab size
train_data = pd.read_csv('train_data.csv')
input_size = len(count_of_csv(train_data))+2

#initialsing the RNN model
model = RNNMod(input_size,100,2,2)

#Initalising the optimizers
optimizer = optim.SGD(model.parameters(),lr = 0.1)

#Initalising the loss function
criterion = nn.CrossEntropyLoss()

iteration = []
accuracy = []

model.train()
#training of the model
for epoch in range(100):
    total_loss = 0
    count = 0
    for i,(review,label) in enumerate(train_dataloader):
        review = torch.transpose(review,0,1)
        review = Variable(review)
        label = Variable(label)
        #clear the gradients
        optimizer.zero_grad()
        #Forward Progation
        output = model(review)
        #calculating the loss
        loss = criterion(output,label)
        #calculating the gradients
        loss.backward()
        #updating the parameters
        #if need use grad clipping or else it is optional
        optimizer.step()
        total_loss += loss.item()
        iteration.append(epoch)
        count += 1
    training_loss = total_loss/count
    print('Epochs:{}  loss: {}'.format(epoch, training_loss))


