import numpy as np
from ordered_set import OrderedSet

def dataset_transform(filename):
    rel_set, node_set = OrderedSet(), OrderedSet()
    with open(filename) as fp:
        file = fp.readlines()
    for line in file:
        sub, rel, obj = line.strip().split('\t')
        node_set.add(sub)
        node_set.add(obj)
        rel_set.add(rel)
    node2id = {node: idx for idx, node in enumerate(node_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

    rows = []
    for line_ in file:
        list = line_.strip().split('\t')
        subject, object = node2id[list[0]], node2id[list[2]]
        relation = rel2id[list[1]]
        row = str(subject) + '\t' + str(object) + '\t' + str(relation) + '\n'
        rows.append(row)
    return rows, node2id


def read_dblp_graph(graph_filename):
    # p -> a : 0
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    #graph_filename = '../data/dblp/dblp_triple.dat'

    relations = set()
    need_nodes = OrderedSet()
    nodes = OrderedSet()
    graph = {}

    # with open(graph_filename) as infile:
    infile, node2id = dataset_transform(graph_filename)
    for line in infile:
        source_node, target_node, relation = line.strip().split('\t')
        source_node = int(source_node)
        target_node = int(target_node)
        relation = int(relation)

        nodes.add(source_node)
        need_nodes.add(source_node)
        nodes.add(target_node)
        relations.add(relation)

        if source_node not in graph:
            graph[source_node] = {}

        if relation not in graph[source_node]:
            graph[source_node][relation] = []

        graph[source_node][relation].append(target_node)

    n_node = len(nodes)
    #print relations
    return node2id, nodes, need_nodes, n_node, len(relations), graph

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def get_dict_key(dic, value):
    keys = list(dic.keys())
    values = list(dic.values())
    idx = values.index(value)
    key = keys[idx]
    return key

# def read_embeddings(filename, n_node, n_embed):
#
#     embedding_matrix = np.random.rand(n_node, n_embed)
#     i = -1
#     with open(filename) as infile:
#         for line in infile.readlines()[1:]:
#             i += 1
#             emd = line.strip().split()
#             embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
#     return embedding_matrix

if __name__ == '__main__':
    node2id, nodes, need_nodes, n_node, n_relation, graph = read_dblp_graph()

    #embedding_matrix = read_embeddings('../data/dblp/rel_embeddings.txt', 6, 64)
    print(graph[1][1])
