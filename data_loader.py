# -*- coding: utf-8 -*-

#importing dependencies
import pandas as pd
from torch.utils.data import Dataset
from data_preprocessing import word_index,pad_data
import torch   

#data importing class
class imdb_dataset(Dataset):
    '''
    This class loads the dataset
    '''
    def __init__(self,path_csv):
        self.df = pd.read_csv(path_csv,encoding="latin-1")
        self.word_to_index = word_index(self.df)
        self.df['review'] = self.df['review'].apply(self.indexify)
        self.df['review'] = self.df['review'].apply(pad_data)
        #self.df['review'] = torch.Tensor(self.df['review'])
        #self.df['label'] = torch.Tensor(self.df['label'].values)
        
    def __getitem__(self,idx):
        x_data = self.df.review[idx]
        y_data = self.df.label[idx]
        return x_data,y_data
    def __len__(self):
        return self.df.shape[0]
    
    def indexify(self,lst_text):
        '''
        This function gives index corresponds to the words
        Input: text in a list
        Output : sequence in a list
        '''
        indices = []
        for word in lst_text:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
            else:
                indices.append(self.word_to_index['__UNK__'])
        return indices