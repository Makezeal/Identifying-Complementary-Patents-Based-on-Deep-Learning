# -*- coding: utf-8 -*-
# @Time : 2023/4/18 12:36
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : Complementary relationship annotation.py
# @Project : pythonProject
import itertools
from functools import reduce

import pandas as pd
import re
import numpy as np
import csv

IPC_infor = pd.read_csv('D:/Initial data and vector results/patent_2022.csv', usecols=['publication_number', 'section_class_subclass_groups'])
IPC = IPC_infor.copy()
IPC['data_ipc'] = IPC['section_class_subclass_groups'].map(lambda x: re.split(r',', x))
# Divide and process IPC classification numbers based on subcategories and large groups
def ipc_split(ipcs):
    ipc_sp = []
    for ipc in ipcs:
        ipc_process = []
        subclass = ipc.split(' ')[0] # Using the first 4 digits of the IPC classification number space to determine subcategories
        group = ipc.split(' ')[1].split('/')[0] # Using symbols/identifying large groups
        if [subclass, group] not in ipc_process:
            ipc_sp.append([subclass, group])
    return ipc_sp
IPC['pre_IPC'] = IPC['data_ipc'].map(lambda x: ipc_split(x))

# Generate a complementary relationship matrix based on IPC classification number, with complementarity of 1, otherwise it is 0
patent = IPC['publication_number'].tolist() # Obtain Patent List
# ipcs = IPC['pre_IPC'].tolist() # Obtain patent IPC list
df = pd.DataFrame(index=patent, columns=patent) # Create a new matrix for storing complementary relationship results
dict = dict(zip(IPC['publication_number'], IPC['pre_IPC'])) # Package patents and their corresponding IPC classification numbers into a dictionary

# A function for determining whether patents are complementary to each other
def if_complementary(a, b):
    result = 0
    ipc_a = dict[a]
    ipc_b = dict[b]
    concat_ipc = [(a_ipc, b_ipc) for a_ipc in ipc_a for b_ipc in ipc_b]
    for ipcs in concat_ipc:
        if ipcs[0][0] == ipcs[1][0] and ipcs[0][1] != ipcs[1][1]: # Different large groups under the same subcategory are complementary
            result += 1
        else:
            result += 0
    if result == 0:
        return 0
    else: # As long as there is a set of IPC classification numbers that are complementary, it is considered complementary
        return 1


with open('D:/Initial data and vector results/Patent complementarity label matrix.csv', 'w', newline='') as file:
    # Create CSV write object
    csv_writer = csv.writer(file)

    # Write header to CSV file
    patent.insert(0, '')
    csv_writer.writerow(patent[:])

    for patent_a in patent[1:]:
        row = []
        row.append(patent_a)
        for patent_b in patent[1:]:
            if patent_a == patent_b:
                row.append(0)
            else:
                row.append(if_complementary(patent_a, patent_b))
        # Write data line by line
        csv_writer.writerow(row)
