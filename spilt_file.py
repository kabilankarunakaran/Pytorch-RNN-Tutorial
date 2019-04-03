# -*- coding: utf-8 -*-

import pandas as pd 

imdb_data = pd.read_csv('/home/lumina/Demo/imdb_master.csv',encoding="latin-1")
#Removing the non tagged data from the dataset
imdb_data = imdb_data[imdb_data.label != 'unsup']
imdb_data = imdb_data.drop(['Unnamed: 0','file'],axis=1)

#Seperating the training and test data
train_data = imdb_data[imdb_data.type =='train']
test_data = imdb_data[imdb_data.type =='test']

#droping the column from dataset
train_data = train_data.drop(['type'],axis =1)
test_data = test_data.drop(['type'],axis =1)

train_data['label'] = train_data['label'].map({'neg':0,'pos':1})
test_data['label'] = test_data['label'].map({'neg':0,'pos':1})

train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')