# -*- coding: utf-8 -*-
# @Time : 2023/9/14 9:31
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : gen_dis_mean.py
# @Project : pythonProject
import pandas as pd
import numpy as np

# 读取第一个CSV文件
df1 = pd.read_csv('patent_gen.csv', names=['patent_id', 'patent_emb'])
df1['patent_emb'] = df1['patent_emb'].map(lambda x: eval(x))

# 读取第二个CSV文件
df2 = pd.read_csv('patent_dis.csv', names=['patent_id', 'patent_emb'])
df2['patent_emb'] = df2['patent_emb'].map(lambda x: eval(x))

# 合并两个DataFrame，使用专利编号进行匹配
merged_df = pd.merge(df1, df2, on='patent_id')
merged_df['mean_emb'] = merged_df.apply(lambda row: np.mean([row['patent_emb_x'], row['patent_emb_y']], axis=0), axis=1)
merged_df['mean_emb'] = merged_df['mean_emb'].map(lambda x: x.tolist())
# 将结果保存到新的CSV文件
merged_df[['patent_id', 'mean_emb']].to_csv('专利向量_hegan.csv', index=False, header=False)

