# -*- coding: utf-8 -*-

import re
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer

#class preprocessing_data(object):
#    def __init__(self,path):
#        self.data = pd.read_csv(path) 
#        #self.word_to_index = self.word_index(self.data)
        
def cleaning_imdb(text):
    '''
    This function turns your text into lowercaser
    Removes stop words from your text 
    Keep only the aplhabets in your text
    Input : String
    Output : String
    '''
    ps = PorterStemmer()
    stopwords_eng = set(stopwords.words("english"))
    imdb_text = text.lower()
    imdb_text = re.sub("[^a-z]"," ",imdb_text)
    split_words = [word for word in imdb_text.split() if word not in stopwords_eng]
    stem_words = [ps.stem(word) for word in split_words]
    imdb_text = " ".join(stem_words)
    return imdb_text

def tokenize(text):
    '''
    This function tokenize the content using space
    Input: text in str format
    Return: list format
    '''
    return [x.lower() for x in text.split()]

def count_of_csv(df,tokenize = tokenize):
    '''
    To count the number of unique words in the file
    Input: dataframe
    Output: dict
    '''
    df['review'] = df['review'].apply(cleaning_imdb)
    df['review'] = df['review'].apply(tokenize)
    word_count = Counter([tok for row in df['review'].values.tolist() for tok in row])
    return word_count

def word_index(df,minimum_count =1, padding_marker = '__PADDING__',unknown_marker = '__UNK__'):
    '''
    Creates a word to index mapping
    Input: 
        df: dataframe
        padding_marker: padding mark
        unknown marker: unknown word marker
        minimum_count : for truncation
    Return : dict
    '''
    counts = count_of_csv(df)
    _word_count = filter(lambda x: minimum_count <=x[1],counts.items())
    tokens = list(zip(*_word_count))[0]
    word_to_index = { tkn: i for i, tkn in enumerate([padding_marker, unknown_marker] + sorted(tokens))}
    return word_to_index

def pad_data(text):
    '''
    This function does zero padding for the sequences
    Input: sequence of numbers
    Output: Sequence Padded to 256 length
    '''
    maxlen = 256
    padded = np.zeros((maxlen,), dtype=np.int64)
    if len(text) > maxlen: padded[:] = text[:maxlen]
    else: padded[:len(text)] = text
    return padded