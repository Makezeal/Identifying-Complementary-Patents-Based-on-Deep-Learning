# -*- coding: utf-8 -*-
# @Time : 2023/7/30 10:56
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于LSTM的专利篇章表示.py
# @Project : pythonProject

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
import torchtext.data
import torchtext.vocab
import csv


# It is the code running in the GPU environment

# read data
def load_csv(datapath):
    file = pd.read_csv(datapath, encoding='utf-8',
                       usecols=['publication_number', 'publication_title', 'abstract', 'claims', 'description'])
    pub_num, name, abstract, claims, descript = \
        file['publication_number'], file['publication_title'], file['abstract'], file['claims'], file['description']
    tuples = []
    for i in range(len(pub_num)):
        patent = name[i] + '. ' + abstract[i] + '. ' + descript[i]
        tuple = [(pub_num[i], patent)]
        tuples += tuple
    df_pos = pd.DataFrame(tuples, columns=['pub_num', 'content'])
    print('Reading data completed')
    return df_pos


# Prepare English text data
data = load_csv('D:/3 Text representation learning/patent_2022.csv')
data = data.iloc[:100]
# data = load_csv('/Initial data and vector results/patent_2022.csv')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
patents = data['pub_num'].tolist()
texts = data['content'].tolist()
tokenized_texts = [tokenizer(text) for text in texts]

# Build a vocabulary list. Define the minimum word frequency, where setting it to 100 means that the vocabulary list must contain at least two words
my_vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_texts, min_freq=100, specials=['<unk>', '<pad>'])
print('Building vocabulary completed')
# Convert text to an index sequence
# sequences = [torch.tensor([my_vocab[word] for word in text]) for text in tokenized_texts]
sequences = [torch.tensor([my_vocab[word] if word in my_vocab else my_vocab["<unk>"] for word in text]) for text in
             tokenized_texts]
print('Convert text to index completed')
# Fill the sequence to the same length
max_sequence_length = 10
padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
padded_sequences = padded_sequences[:, :max_sequence_length]

# Creating an LSTM autoencoder model
embedding_dim = 100  # Embedding layer dimension
lstm_units = 128  # Number of LSTM layer units

print('LSTM model creation completed')
class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(len(my_vocab), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(lstm_units, embedding_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim, len(my_vocab))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

encoder = LSTMEncoder()
decoder = LSTMDecoder()

# Encoding and decoding text
encoded_texts = encoder(padded_sequences)
decoded_texts = decoder(encoded_texts)
print('Text encoding and decoding completed')
# Extract text vectors
text_vectors = encoded_texts.detach().cpu().numpy()
print('Extract text vector completed')
# Write the patent vector and its ID together into the dataframe
tuples = []
for i in range(len(patents)):
    patent_id, text_emb = patents[i], text_vectors[i].tolist()
    tuples.append((patent_id, text_emb))

# Write to file
df = pd.DataFrame(tuples, columns=['patent_id', 'patent_emb'])
df.to_csv('专利向量_lstm.csv', index=False, header=False)
print('Writing file completed')