# -*- coding: utf-8 -*-
# @Time : 2022/8/26 17:04
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于ESimCSE+self_attention的专利篇章表示.py
# @Project : pythonProject
import csv
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers.ESimCSE import ESimCSE
import torch.nn.functional as F
print('Successfully imported library model')

# Need to run with VPN enabled

# To avoid situations where there is no cuda and running errors occur
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
model_name = 'bert-large-uncased' # You need to download it yourself, as there are resources available online
train_batch_size = 50
num_epochs = 4
max_seq_length = 64


# Model save path
model_save_path = 'stsb_esimcse_bert_pub'

#
# Modeling
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
        try:
            patent[pub] = name[i] + '. ' + abstract[i] + '. ' + descript[i]
        except:
            pass
    print('Successfully read data')
    return patent


# Segmenting sentences and converting them into vector matrices separately
def sentences_represent(datapath):
    patent = load_csv(datapath)
    print('Load_csv successful')
    for key, value in patent.items():
        sentence_representation = []
        sentences = sent_tokenize(value)
        for sentence in sentences:
            sentence_vector = model.encode(sentence.replace('\n', ' '))
            sentence_representation.append(sentence_vector)
        sen_array = np.stack(sentence_representation, axis=0)
        # Convert to tensor form
        tensor_01 = torch.from_numpy(sen_array)
        # Updimensionality of tensors
        tensor_02 = tensor_01.unsqueeze(0)
        patent[key] = tensor_02
        print('Patient [key] successful')
    return patent


# Self attention mechanism
class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.device = torch.device('cuda:0')
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size, device=self.device)
        self.query_layer = nn.Linear(input_size, hidden_size, device=self.device)
        self.value_layer = nn.Linear(input_size, hidden_size, device=self.device)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.to(device)

        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(torch.tensor(attention_scores), dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        flatten = nn.Flatten()
        tensor = flatten(context)
        linear_1 = nn.Linear(in_features=tensor.size()[1], out_features=768, device=self.device) # 统一维度
        tensor_1 = linear_1(tensor)
        relu_1 = F.relu(tensor_1, inplace=True)
        layer_2 = nn.Linear(in_features=tensor_1.size()[1], out_features=100, device=self.device)  # 统一维度
        result = layer_2(relu_1)
        return result


dict_patent = sentences_represent('D:/Initial data and vector results/patent_2022.csv')

saveDict = {}

fileName = "D:/Initial data and vector results/专利向量_esimcse_attention.csv"
with open(fileName, "a+", newline='', encoding='utf-8') as csv_file:
    for key, value in dict_patent.items():
        attention = selfAttention(2, 1024, 20)
        att_result = attention.forward(value)
        saveDict[key] = att_result

        # Write to file
        writer = csv.writer(csv_file)
        writer.writerow([key, att_result.tolist()])

