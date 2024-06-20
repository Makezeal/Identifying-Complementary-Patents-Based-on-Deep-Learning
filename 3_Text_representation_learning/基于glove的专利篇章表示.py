# -*- coding: utf-8 -*-
# @Time : 2023/7/29 10:21
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于glove的专利篇章表示.py
# @Project : pythonProject
import csv
import re
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords

"""read data"""
def load_csv(datapath):
    file = pd.read_csv(datapath, usecols=['publication_number', 'publication_title', 'abstract', 'claims', 'description'])
    pub_num, name, abstract, claims, descript = \
        file['publication_number'], file['publication_title'], file['abstract'], file['claims'], file['description']
    patent = {}
    for i in range(len(pub_num)):
        pub = pub_num[i]
        try:
            patent[pub] = name[i] + '. ' + abstract[i] + '. ' + descript[i]
        except:
            pass
    return patent


"""Reading Glove word vectors"""
# At the beginning, there was an error message“ValueError: could not convert string to float: '.'”，So I joined try
glove_embedding = {}
f = open('D:/3 Text representation learning/glove.840B.300d.txt', encoding='utf-8') # You need to download resources yourself
try:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embedding[word] = vector
except:
    f.__next__()
f.close()


"""Data preprocessing to obtain word vectors for words and sentence vectors for sentences"""
dict_patent = load_csv('D:/Initial data and vector results/patent_2022.csv')
eng_stopwords = set(stopwords.words('english'))

# Clean text data
def clean_text(text, remove_stopwords=False):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

# Take the Glove word vector
def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.asarray([glove_embedding[w] for w in words if w in glove_embedding], dtype='float32')
    return array.mean(axis=0)


"""Using Glove to represent samples"""
saveDict = {}
fileName = "D:/Initial data and vector results/专利向量_glove.csv"
with open(fileName, "w", newline='') as csv_file:
    for key, value in dict_patent.items():
        dim_300 = to_review_vector(value)
        _tensor = torch.from_numpy(dim_300)
        layer = torch.nn.Linear(in_features=300, out_features=100)  # Unified dimension
        result = layer(_tensor)
        saveDict[key] = result

        # Write to file
        writer = csv.writer(csv_file)
        writer.writerow([key, result.tolist()])