# -*- coding: utf-8 -*-
# @Time : 2023/3/29 17:18
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : 异构网络构建.py
# @Project : pythonProject
import pandas as pd

# read data
def load_csv(datapath):
    file = pd.read_csv(datapath, usecols=['publication_number', 'authors', 'section_class_subclass_groups', 'reference'])
    inventor, ipc, refer = {}, {}, {}
    for index, row in file.iterrows():
        pub_num = row['publication_number']
        inventor[pub_num] = row['authors'].split(',')
        ipc[pub_num] = row['section_class_subclass_groups'].split(',')
        refer[pub_num] = str(row['reference']).split(',')
    return inventor, ipc, refer

# Heterogeneous network construction
def hetnet(datapath):
    inventor, ipc, refer = load_csv(datapath)
    inventor_relate, ipc_relate, refer_relate = [], [], []
    for key, value in inventor.items():
        for single_inventor in value:
            sample = '/m/01 ' + str(key) + '\t' + '/patent/invented by/inventor' + '\t' + '/m/02 ' + str(single_inventor)
            inventor_relate.append(sample)
    for key, value in ipc.items():
        for ipc_num in value:
            sample = '/m/01 ' + str(key) + '\t' + 'patent/include/ipc' + '\t' + '/m/03 ' + str(ipc_num.replace(' ', '')) # If the uspto_XMLparse.py is modified, there is no need to replace it
            ipc_relate.append(sample)
    for key, value in refer.items():
        for patent_num in value:
            sample = '/m/01 ' + str(key) + '\t' + 'patent/cite/patent' + '\t' + '/m/01 ' + str(patent_num)
            refer_relate.append(sample)
    with open('D:/Code/Initial data and vector results/专利异构知识图.txt', 'w', encoding='utf-8') as f:
        for row in inventor_relate:
            f.write(row)
            f.write('\n')
        for row in ipc_relate:
            f.write(row)
            f.write('\n')
        for row in refer_relate:
            f.write(row)
            f.write('\n')
        f.close()
    return inventor_relate, ipc_relate, refer_relate

hetnet('D:/Code/Initial data and vector results/patent_2022.csv')