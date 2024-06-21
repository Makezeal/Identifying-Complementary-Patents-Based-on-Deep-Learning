# -*- coding: utf-8 -*-
# @Time : 2022/10/5 16:26
# @Author : 施佳璐
# @Email : shijialu0716@163.com
# @File : Nonlinear merging of CBAM+CNN with Single Features.py
# @Project : pythonProject
import json
import itertools
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Add, Reshape, Permute, multiply, Lambda, Concatenate
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import keras_metrics as km
from sklearn.metrics import roc_auc_score
import logging

# Configuration logs
logging.basicConfig(filename='/compgcn.log', level=logging.INFO)

# Replacing the print function
def log_print(msg):
    logging.info(msg)
    print(msg)
''' Channel attention mechanism：
    When compressing the spatial dimension of the input feature map, the author not only considered average pooling, but also,
    By introducing max pooling as a supplement, a total of two one-dimensional vectors can be obtained through two pooling functions.
    Global average pooling provides feedback for every pixel on the feature map, while global max pooling
    When performing gradient backpropagation calculations, only the areas with the highest response in the feature map have gradient feedback, which can serve as a supplement to GAP.
'''
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    print('Completed running channel attention mechanism')
    return multiply([input_feature, cbam_feature])


''' Spatial attention mechanism:
    Still use average pooling and max pooling to compress the input feature map,
    However, the compression here has become channel level compression, and the input features have been separately compressed in the channel dimension
    Mean and max operations. Finally, two two-dimensional features were obtained and concatenated together according to the channel dimension
    Obtain a feature map with 2 channels, and then use a hidden layer pair containing a single convolutional kernel
    When performing convolution operations, it is necessary to ensure that the final feature obtained is consistent with the input feature map in the spatial dimension,
'''
def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    print('Completed running spatial attention mechanism')
    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    # Experimental verification shows that the approach of channel after space is more effective than the approach of space after channel or parallel channel space
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )

    return cbam_feature


# modeling
class Complementarity_Model(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nd = tf.Variable([0.0], dtype=tf.float32)
    def model(self, input_shape):
        #Parameter: input_shape - Dimension of input data
        #Return: Model - The model of Keras created
        #1. Define a placeholder for a tensor with the dimension input_shape
        X_input = Input(input_shape)
        X = BatchNormalization(axis=3, name='bn0')(X_input)
        # 2.Use CONV ->BN ->RELU block for X
        X = Conv2D(64, (2, 2), strides=(1, 1), padding='same', name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn1')(X)
        X = Conv2D(64, (2, 2), strides=(1, 1), padding='same', name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn2')(X)
        X = cbam_block(X)
        X = Activation('relu')(X)
        # 3.Maximum pooling layer
        X = MaxPooling2D((2, 2), name='max_pool')(X ** 2)
        X = BatchNormalization(axis=3, name='bn3')(X)
        # 4.Dimensionality reduction, matrix conversion to vectors+fully connected layers
        X = Flatten()(X)
        X = X - self.nd  # Add a complementary threshold
        X = Dense(200, activation='relu', name='dense0')(X)
        X = Dense(1, activation='sigmoid', name='prob')(X)
        model = Model(X_input, X)
        print('Built model')
        return model

# # Custom R ² Indicator Function
# def r_squared(y_true, y_pred):
#     ss_res = K.sum(K.square(y_true - y_pred))
#     ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return 1 - ss_res / (ss_tot + K.epsilon())

# Custom AUC calculation function
def auc(y_true, y_pred):
    auc_score = tf.py_function(roc_auc_score, (y_true, y_pred), tf.float32)
    print('Calculate AUC')
    return auc_score

def model_train_fit(patent_list, labels):
    ## Merge the data to form a two-dimensional vector and its corresponding labels
    input_data = []
    for pat in patent_list:
        a, b = pat[0], pat[1]
        emb_a, emb_b = vectors[a], vectors[b]
        con_vector = [emb_a, emb_b]
        input_data.append(con_vector)
        if len(input_data) % 1000000 == 0:
            log_print(len(input_data))
    x = np.array(input_data)  # input data
    y = np.array(labels)  # Corresponding label
    # classes = np.array(list(set(labels)))  # classification


    # Partition dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print('x:', x)
    print('y:', y)
    print('X_train.shape:',X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*2, int(X_train.shape[2]/2), 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*2, int(X_test.shape[2]/2), 1)
    Y_train = Y_train.reshape((1, Y_train.shape[0])).T
    Y_test = Y_test.reshape((1, Y_test.shape[0])).T
    ## Data related information
    log_print("number of training examples = " + str(X_train.shape[0]))
    log_print("number of test examples = " + str(X_test.shape[0]))
    log_print("X_train shape: " + str(X_train.shape))
    log_print("Y_train shape: " + str(Y_train.shape))
    log_print("X_test shape: " + str(X_test.shape))
    log_print("Y_test shape: " + str(Y_test.shape))

    # Training and testing models: creating model entities, compiling models, training models, evaluating models
    ## 1.Create a model entity
    complementarity_model = Complementarity_Model().model(X_train.shape[1:])
    ## 2.Compilation Model
    # complementarity_model.compile("adam", "binary_crossentropy", metrics=['acc', km.f1_score(), km.binary_precision(), km.binary_recall()])
    complementarity_model.compile("adam", "binary_crossentropy",
                                  metrics=['acc', 'mse', 'mae', km.f1_score(), km.binary_precision(), km.binary_recall()])
    ## 3.Training model
    complementarity_model.fit(X_train, Y_train, epochs=30, batch_size=5000, class_weight={0: 0.3, 1: 1})
    print('Trained model completed')
    ## 4.Evaluation model
    preds = complementarity_model.evaluate(X_test, Y_test, batch_size=3200, verbose=1, sample_weight=None)
    errors = preds[0]
    accuracy = preds[1]
    mse = preds[2]
    mae = preds[3]
    f1_score = preds[4]
    precision = preds[5]
    recall = preds[6]
    print('Model has been evaluated and completed')
    # auc_score = preds[7]
    ## 5.Forecast results
    # prediction = x.reshape((x.shape[0], x.shape[1]*2, int(x.shape[2]/2), 1))
    # result = complementarity_model.predict(prediction)
    return errors, accuracy, mse, mae, f1_score, precision, recall #, result

# Define a function to normalize a single vector
def normalize_vector(vector):
    vector = eval(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

## Obtain input data and its corresponding labels
label = pd.read_csv('/Patent complementarity label matrix.csv', index_col=0, low_memory=False)
total_vector = pd.read_csv('/专利向量_esimcse_attention.csv', index_col=False, names=['patent_num', 'vector'])
print('Data read in')
total_vector['vectors_normalized'] = total_vector['vector'].apply(normalize_vector)
print('total_vector',total_vector)
vectors = total_vector.set_index(['patent_num'])['vectors_normalized'].to_dict()
print('Total_vector successful')


## Obtain a list of all patents
patent = total_vector['patent_num'].tolist()[:1000]
patent_sample = []
label_ = []
for a in patent:
    for b in patent:
        patent_sample.append([a, b])
        label_.append(label[a][b])
print('Successfully obtained patent list')
i, j = 0, 0
sum_error, sum_acc, sum_mse, sum_mae, sum_f1, sum_precision, sum_recall = 0, 0, 0, 0, 0, 0, 0
output = []
while j + 1000000 < len(patent_sample):
    i += 1
    log_print('第{}次循环'.format(i))
    errors, accuracy, mse, mae, f1_score, precision, recall = model_train_fit(patent_sample[j:j+1000000], label_[j:j+1000000])
    sum_error += errors
    sum_acc += accuracy
    sum_mse += mse
    sum_mae += mae
    sum_f1 += f1_score
    sum_precision += precision
    sum_recall += recall
    # output.append(result)
    j += 1000000
    print('Loop successful')

errors_, accuracy_, mse_, mae_, f1_score_, precision_, recall_ = model_train_fit(patent_sample[j-1000000:], label_[j-1000000:])
print('Calculation accuracy successful')
avg_error = (sum_error+errors_)/(i+1)
avg_acc = (sum_acc+accuracy_)/(i+1)
avg_mse = (sum_mse+mse_)/(i+1)
avg_mae = (sum_mae+mae_)/(i+1)
avg_f1 = (sum_f1+f1_score_)/(i+1)
avg_precision = (sum_precision+precision_)/(i+1)
avg_recall = (sum_recall+recall_)/(i+1)
# output.append(result_)

log_print("avg_loss = " + str(avg_error))
log_print("avg_acc = " + str(avg_acc))
log_print("avg_mse = " + str(avg_mse))
log_print("avg_mae = " + str(avg_mae))
log_print("avg_f1 = " + str(avg_f1))
log_print("avg_precision = " + str(avg_precision))
log_print("avg_recall = " + str(avg_recall))