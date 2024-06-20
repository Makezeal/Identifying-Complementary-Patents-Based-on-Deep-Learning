# -*- coding: utf-8 -*-
# @Time : 2022/8/26 17:04
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于ESimCSE的专利篇章表示.py
# @Project : pythonProject
import csv
import re
import numpy as np
import pandas as pd
import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers.ESimCSE import ESimCSE

# Need to run with VPN enabled

# To avoid situations where there is no cuda and running errors occur
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
model_name = 'D:/3 Text representation learning/bert-large-uncased'
train_batch_size = 50
num_epochs = 4
max_seq_length = 64


# 模型保存路径
model_save_path = 'D:/3 Text representation learning/stsb_esimcse_macbert_pub'


# 建立模型
moco_encoder = SentenceTransformer(model_name).to(device)
moco_encoder.__setattr__("max_seq_length", max_seq_length)
word_embedding_model = models.Transformer(model_name)
word_embedding_model.__setattr__("max_seq_length", max_seq_length)
pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode="cls",
                               pooling_mode_cls_token=True)
model = ESimCSE(modules=[word_embedding_model, pooling_model], dup_rate=0.32, q_size=150)
print('Successfully established SentenceTransformer model')

# Save the model

model.save(model_save_path)


# read data
def load_csv(datapath):
    file = pd.read_csv(datapath, usecols=['publication_number', 'publication_title', 'abstract', 'claims', 'description'])
    pub_num, name, abstract, claims, descript = \
        file['publication_number'], file['publication_title'], file['abstract'], file['claims'], file['description']
    patent = {}
    for i in range(len(pub_num)):
        pub = pub_num[i]
        patent[pub] = name[i] + '. ' + abstract[i] + '. ' + descript[i]
    print('Successfully read data')
    return patent


# Segmenting sentences and converting them into vector matrices separately
def sentences_represent(datapath):
    patent = load_csv(datapath)
    print('Load_scv successful')
    for key, value in patent.items():
        sentence_representation = []
        sentences = []
        index = 0
        while index + 512 < len(value):
            sentences.append(value[index:index + 512])
            index += 512
        for sentence in sentences:
        # value = re.split(r'.', value)
        # for sentence in value:
            sentence_vector = model.encode(sentence)
            sentence_representation.append(sentence_vector)
        patent[key] = np.stack(sentence_representation, axis=0)
        print('Patient [key] successful')
    return patent


# Using Convolutional Neural Networks to Obtain Patent Text Representation
def CNN_folding(array):
    # Convert to tensor form
    tensor_01 = torch.from_numpy(array)
    # print(tensor_01)
    # Updimensionality of tensors
    tensor_02 = tensor_01.unsqueeze(0)
    # Coordinate System Conversion
    tensor = tensor_02.permute(0, 2, 1)
    # Convolutional kernels with a size of 2
    conv2 = nn.Conv1d(in_channels=1024, out_channels=300, kernel_size=3)
    output_01 = conv2(tensor)
    # Get dimension (3) for maximum pooling
    size = output_01.shape[2]
    # Maximizing pooling to keep the third dimension consistent at 1
    max_pool = nn.MaxPool1d(size, stride=2)
    output_02 = max_pool(output_01)
    # print(output_02.shape)
    # Using one-dimensional convolution as a fully connected layer
    conv1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=1)
    output_03 = conv1(output_02)
    # Convert the result into a one-dimensional tensor
    tensor = output_03.squeeze()
    output = tensor.detach().numpy()
    return output


dict_patent = sentences_represent('D:/Initial data and vector results/patent_2022.csv')
saveDict = {}
for key, value in dict_patent.items():
    conv_result = CNN_folding(value)
    print('CNN_folding successful')
    saveDict[key] = conv_result


# 写入文件中
fileName = "D:/Initial data and vector results/专利向量_esimcse.csv"
with open(fileName, "a+", newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in saveDict.items():
        writer.writerow([key, value.tolist()])
