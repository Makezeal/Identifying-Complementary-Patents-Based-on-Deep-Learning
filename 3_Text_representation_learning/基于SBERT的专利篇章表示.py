# -*- coding: utf-8 -*-
# @Time : 2022/8/26 16:36
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于SBERT的专利篇章表示.py
# @Project : pythonProject
# 利用BERT的预训练模型获得句子向量

import csv
import nltk
nltk.download('punkt')  # Download punkt data to support clauses
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import torch
from torch import nn
from itertools import chain
from sentence_transformers import SentenceTransformer # This sentence.transformers is an installed package, not the software package in the data processing folder

# Need to run with VPN enabled


model = SentenceTransformer('bert-base-nli-mean-tokens')

# Segmenting sentences and converting them into 100 dimensional vectors separately
datapath = 'patent_2022.csv'

file = pd.read_csv(datapath,usecols=['publication_number', 'publication_title', 'abstract', 'claims', 'description'])
pub_num, name, abstract, claims, descript = \
    file['publication_number'], file['publication_title'], file['abstract'], file['claims'], file['description']
patent = {}
for i in range(len(pub_num)):
    pub = pub_num[i]
    patent[pub] = name[i] + '. ' + abstract[i] + '. ' + descript[i]



a = 0
for key, value in patent.items():
    a+=1
    print('Completed running data in ',a,'rows')
    sentence_representation = []
    sentences = sent_tokenize(value)
    for sentence in sentences:
        sentence_vector = model.encode(sentence)
        sentence_representation.append(sentence_vector)
    # Convert to numpy array along the first axis (axis 0)
    sen_array = np.stack(sentence_representation, axis=0)
    # Calculate the average value along the first axis (axis 0)
    sen_vector = np.mean(sen_array, axis=0)
    # Convert to PyTorch tensor
    sen_vector_tensor = torch.tensor(sen_vector, dtype=torch.float32)
    # Define a linear layer to map (1768) dimensions to (1100) dimensions
    linear_layer = nn.Linear(768, 100)
    # Using linear layers for mapping
    output_vector = linear_layer(sen_vector_tensor)
    result = output_vector.detach().numpy().tolist()
    patent[key] = result

patent = pd.DataFrame(patent)
print(patent.head(10))
print(patent.info())
saveDict = patent

# Write to file
fileName = "专利向量_SBERT.csv"
with open(fileName, "w", newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in saveDict.items():
        writer.writerow([key, value])
