#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:56:33 2019

@author: lumina
"""
#importing the dependcies
from data_loader import imdb_dataset
from torch.utils.data import DataLoader
from data_preprocessing import word_index
from RNN_Module import RNNMod
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import pandas as pd
 
#Insitantiate the dataloading class    
train_loader = imdb_dataset('train_data.csv')
test_loader = imdb_dataset('test_data.csv')

# load the dataset in batches using pytorch Dataloader in built class
train_dataloader = DataLoader(dataset = train_loader, batch_size = 1024, shuffle = True, num_workers = 4)
test_dataloader = DataLoader(dataset = test_loader, batch_size = 1024, shuffle = True, num_workers = 4)

#Taking the vocab size
train_data = pd.read_csv('train_data.csv')
input_size = word_index(train_data,5)
input_vocab = len(input_size)

#initialsing the RNN model
model = RNNMod(input_vocab,100,2,2)

#Initalising the optimizers
optimizer = optim.RMSprop(model.parameters(),lr = 0.05)

#Initalising the loss function
criterion = nn.CrossEntropyLoss()


def train_model(model,train_data,epoch,optimizer,criterion):
    '''
    This function trains the model
    Args: Training data in the form of batches
    Output: returns the model
    '''
    iteration = []
    model.train()
    #training of the model
    #for epoch in range(100):
    total_loss = 0
    count = 0
    for i,(review,label) in enumerate(train_data):
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
        #if needed use grad clipping or else it is optional
        optimizer.step()
        total_loss += loss.item()
        iteration.append(epoch)
        count += 1
    training_loss = total_loss/count
    #print('Epochs:{}  loss: {}'.format(epoch, training_loss))
    return training_loss

def eval_model(model,test_data,criterion):
    '''
    This function trains the model
    Args: Training data in the form of batches
    Output: returns the model
    '''
    eval_loss = 0
    total = 0
    correct = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for test_review,test_label in test_data:
            test_review = torch.transpose(test_review,0,1)
            prediction = model(test_review)
            pred_idx = torch.max(prediction,1)[1]
            loss = criterion(prediction,test_label)
            eval_loss += loss.item()
            total += test_label.size(0)
            correct += (pred_idx == test_label).sum()
        accuracy = 100 * correct / float(total)
    return eval_loss,accuracy

for epoch in range(5):
    training_loss = train_model(model,train_dataloader,epoch,optimizer,criterion)
    eval_loss,eval_acc = eval_model(model,test_dataloader,criterion)
    print('Epoch:{}  train_loss: {}  val_loss: {}  val_acc: {}'.format(epoch, training_loss,eval_loss,eval_acc))
    
#Model Saving and reloading for inference
model_train = train_model(train_dataloader)
torch.save(model_train.state_dict(),'imdb-model.pt')

