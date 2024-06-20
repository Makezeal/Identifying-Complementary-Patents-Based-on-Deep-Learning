# -*- coding: utf-8 -*-
# @Time : 2022/8/24 14:49
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 基于doc2vec的专利篇章表示.py
# @Project : pythonProject
import jieba
import pandas as pd
from collections import namedtuple
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
import csv
import re


# read data
def load_csv(datapath):
    file = pd.read_csv(datapath, usecols=['publication_number', 'publication_title', 'abstract', 'claims', 'description'])
    pub_num, name, abstract, claims, descript = \
        file['publication_number'], file['publication_title'], file['abstract'], file['claims'], file['description']
    patent = {}
    for i in range(len(pub_num)):
        pub = pub_num[i]
        patent[pub] = name[i] + '. ' + abstract[i] + '. ' + descript[i]
    return patent


# Create a list of stop words
eng_stopwords = set(stopwords.words('english'))
# Clean text data and segment sentences into words
def seg_sentence(text, remove_stopwords=False):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    def __str__(self):
        return '%s<%s, %s>' % (self.__class__.__name__, self.words, self.tags)


def sentence_tag(datapath):
    patent_dict = load_csv(datapath)
    patent = list(patent_dict.values())
    tagged_data = []
    for i in range(len(patent)):
        patent[i] = seg_sentence(patent[i], remove_stopwords=True)
        data = TaggedDocument(words=patent[i], tags=[str(i)])
        tagged_data.append(data)
    return tagged_data


# print(sentence_tag('data/patent_2022.csv'))


# model training

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)
datapath = 'D:/Initial data and vector results/patent_2022.csv'
tagged_data = sentence_tag(datapath)
model.build_vocab(tagged_data)


for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
dict_patent = load_csv(datapath)


saveDict = {}
fileName = "D:/Initial data and vector results/专利向量_doc2vec.csv"
with open(fileName, "w", newline='') as csv_file:
    patent_list = list(dict_patent.keys())
    for i in range(len(dict_patent)):
        pub_num = patent_list[i]
        vector = model.docvecs[str(i)]
        saveDict[pub_num] = vector

        # Write to file
        writer = csv.writer(csv_file)
        writer.writerow([pub_num, vector.tolist()])