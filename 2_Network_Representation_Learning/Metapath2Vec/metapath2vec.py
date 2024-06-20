# -*- coding: utf-8 -*-
# @Time : 2023/9/5 15:12
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : metapath2vec.py
# @Project : pythonProject
import pandas as pd
import numpy as np
import dgl
import scipy.sparse as spp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
# Attention: Due to memory limitations during runtime, a device with 24GB of graphics memory is used (4 yuan/hour for Moment Pool Cloud)

total_patent = []
# Create three dictionaries to store the relationships between patents
inventor_relations = {}
ipc_relations = {}
cite_relations = {}
# Read data from a text file and process it
with open('专利异构知识图.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        patent, relation, entity = line.strip().split('\t')
        if relation == '/patent/invented by/inventor':
            inventor = entity
            # If there is a relationship between the inventor and the patent, add it to the dictionary
            if inventor not in inventor_relations:
                inventor_relations[inventor] = set()
            inventor_relations[inventor].add(patent)
            patent_ = patent.replace('/m/01 ', '')
            if patent_ not in total_patent:
                total_patent.append(patent_)
        elif relation == 'patent/include/ipc':
            ipc = entity
            # If there is a relationship between the inventor and the patent, add it to the dictionary
            if ipc not in ipc_relations:
                ipc_relations[ipc] = set()
            ipc_relations[ipc].add(patent)
        elif relation == 'patent/cite/patent':
            cite = entity
            # If there is a relationship between the inventor and the patent, add it to the dictionary
            if cite not in cite_relations:
                cite_relations[cite] = set()
            cite_relations[cite].add(patent)

def patent_matrix(relations):
    # Obtain all different patents
    patents = list(set(patent for patents in relations.values() for patent in patents))
    patents.sort()  # Sort patents to ensure consistent order in the matrix

    # Create a bipartite matrix between patents, initialize to 0
    matrix = [[0] * len(patents) for _ in range(len(patents))]

    # Fill the matrix according to the relationship dictionary
    for inventor, patents in relations.items():
        patents = list(patents)
        for i in range(len(patents)):
            for j in range(i + 1, len(patents)):
                patent1 = patents[i]
                patent2 = patents[j]

                # If the inventors share a patent, set the corresponding position in the matrix to 1
                matrix[patents.index(patent1)][patents.index(patent2)] = 1
                matrix[patents.index(patent2)][patents.index(patent1)] = 1  # symmetric matrix
    coo_graph = spp.coo_matrix(matrix)
    graph = dgl.from_scipy(coo_graph)
    return graph

sum_patent = len(total_patent)
inventor_graph = patent_matrix(inventor_relations)
ipc_graph = patent_matrix(ipc_relations)
cite_graph = patent_matrix(cite_relations)

hete_graph = dgl.heterograph({
    ('patent', 'co_inventor', 'inventor'): inventor_graph.edges(),
    ('patent', 'co_ipc', 'ipc'): ipc_graph.edges(),
    ('patent', 'co_cite', 'patent'): cite_graph.edges()
})

def positive_sampler(path):
    '''
        Create a window sized sliding window for each path,
        For example, 0 1 2 3 4, with a window size of 2, returns pos_u=[0,0,1,1...], pos_v=[1,2,0,2,3...]
    '''
    pos_u, pos_v = [], []
    for i in range(len(path)):
        if len(path) == 1:
            continue
        u = path[i]
        v = np.concatenate([path[max(i-window, 0):i], path[i+1:i+window+1]], axis=0)
        pos_u.extend([u]*len(v))
        pos_v.extend(v)
    return pos_u, pos_v

def get_negative_ratio(metapath):
    '''
        Establish a negative ratio for all nodes based on their frequency of occurrence. The higher the frequency of occurrence, the more likely it is to appear in negative sampling
        The returned ratio is the probability of each node being negatively sampled
    '''
    node_frequency = dict()
    sentence_count, node_count = 0, 0
    for path in metapath:
        for node in path:
            node_frequency[node] = node_frequency.get(node, 0)+1
            node_count += 1
    pow_frequency = np.array(list(map(lambda x: x[-1], sorted(node_frequency.items(), key=lambda asd: asd[0]))))**0.75
    node_pow = np.sum(pow_frequency)
    ratio = pow_frequency / node_pow
    return ratio

def negative_sampler(path, ratio, nodes):
    '''
    Perform negative sampling based on the probability table ratio from the previous function to negative sampling
    '''
    negtives_size = 5
    negatives = []
    while len(negatives) < 5:
        temp=np.random.choice(nodes, size=negtives_size-len(negatives), replace=False, p=ratio)
        negatives.extend([node for node in temp if node not in path])
    return negatives

def create_node2node_dict(graph):
    src_dst = {}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        src, dst = src.item(), dst.item()
        if src not in src_dst.keys():
            src_dst[src] = []
        src_dst[src].append(dst)
    return src_dst
window = 2 # Here is the window size when taking the metapath
metapaths = [] # All Metapaths
num_walks = 10 # How many times each node runs
walk_len = 100 # The length of each path
metapath_type = ['coinventor', 'coipc', 'cocite'] # According to the paper, the author used AVAT

edge_per_graph = {} # Create a dictionary for each graph, where the key is the node number and the value is the node number that the key can reach in the graph
edge_per_graph['coinventor'] = create_node2node_dict(inventor_graph)
edge_per_graph['coipc'] = create_node2node_dict(ipc_graph)
edge_per_graph['cocite'] = create_node2node_dict(cite_graph)


def Is_isolate(node):
    for rel in metapath_type:
        if node in edge_per_graph[rel].keys():
            return 0
    return 1

for walk in tqdm(range(num_walks)):
    for cur_node in list(range(sum_patent)):
        stop = 0
        path = []
        path.append(cur_node)
        while len(path) < walk_len and stop == 0:
            for rel in metapath_type:
                if len(path) == walk_len or Is_isolate(cur_node):
                    stop = 1
                    break
                if edge_per_graph[rel].get(cur_node, -1) == -1:
                    continue

                cand_nodes = edge_per_graph[rel][cur_node]

                cur_node = np.random.choice(cand_nodes, size=1)[0]
                path.append(cur_node)
        metapaths.append(path)

pos_us, pos_vs, neg_vs = [], [], []
nodes = list(range(sum_patent))
ratio = get_negative_ratio(metapaths)
for path in metapaths:
    pos_u, pos_v = positive_sampler(path)
    for u, v in zip(pos_u, pos_v):
        negative_nodes = negative_sampler(path, ratio, nodes)
        neg_vs.append(negative_nodes)
    pos_us.extend(pos_u)
    pos_vs.extend(pos_v)
pos_us = torch.LongTensor(pos_us)
pos_vs = torch.LongTensor(pos_vs)
neg_vs = torch.LongTensor(neg_vs)

# Simple metapath2vec
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension).cuda()
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension).cuda()

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        # Move data to GPU
        pos_u = pos_u.cuda()
        pos_v = pos_v.cuda()
        neg_v = neg_v.cuda()

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

skip_model = SkipGramModel(sum_patent, 100)
optimizer = torch.optim.Adam(skip_model.parameters(), lr=0.001)
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    loss = skip_model(torch.tensor(pos_us), torch.tensor(pos_vs), torch.tensor(neg_vs))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print('epoch {0}  loss {1}'.format(epoch, loss))

embedding = skip_model.u_embeddings.weight.cpu().data.numpy()
# Convert each row of the array to a Series, with the first column using list data and the second column using array row data
data = {'Column1': total_patent, 'Column2': [row for row in embedding]}
# Create a DataFrame
df = pd.DataFrame(data)
df['Column3'] = df['Column2'].apply(lambda x: x.tolist())
df[['Column1', 'Column3']].to_csv('专利向量_metapath2vec.csv', header=False, index=False)