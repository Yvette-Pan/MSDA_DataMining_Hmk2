#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:07:13 2018

@author: yuranpan
"""

import os
#os.chdir('/Users/yuranpan/Desktop/Fordham/Data_Mining/Hmk2/Hmk2')

# read csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

spam_train = pd.read_csv('spam_train.csv', encoding = 'utf-8')
spam_test = pd.read_csv('spam_test.csv')

#############################
#Implement the KNN Classifier
#############################

def attributes(dataset, starting_col):
    
    attributes = np.array(dataset.iloc[:,starting_col:-1])
    
    return attributes

def label(dataset):
    label = np.array(dataset.iloc[:,-1])
    return label

attributes_train = attributes(spam_train, 0)
attributes_test = attributes(spam_test, 1)
label_train = label(spam_train)
label_test = label(spam_test)


def label_based_on_neighbors(label_train, indexes_neighbor):
    label_neighbor_list = list()
    for l in indexes_neighbor:
        label_neighbor = label_train[l]
        label_neighbor_list.append(label_neighbor)
    count_1 = np.count_nonzero(label_neighbor_list)
    count_0 = len(label_neighbor_list) - count_1
    if count_1 > count_0:
        label_test_predict = 1
    else:
        label_test_predict = 0
    return label_test_predict 
   


def KNN_classifier(attributes_train,label_train, attributes_test, k):
    size_train = attributes_train.shape[0]
    size_test = attributes_test.shape[0]
    label_test_predict_list = list()
    for test in range(0, size_test):
        euclidean_distance_all = np.array([])
        for train in range(0, size_train):
            euclidean_distance = np.sqrt(sum((attributes_train[train] - attributes_test[test]) ** 2))
            euclidean_distance_all = np.append(euclidean_distance_all, euclidean_distance)
        indexes = euclidean_distance_all.argsort()[:k]            
        label_test_predict_list.append(label_based_on_neighbors(label_train, indexes))
    return label_test_predict_list




def accuracy(attributes_train, label_train, attributes_test, label_test, k):
    label_test_predict_list = KNN_classifier(attributes_train, label_train, attributes_test, k)
    temp_diff = label_test_predict_list - label_test
    sample_size = temp_diff.shape[0]
    count_correct = sample_size - np.count_nonzero(temp_diff)
    accuracy= count_correct/sample_size
    return accuracy

# 1(a) test accuracy without normalized features
attributes_train = attributes(spam_train, 0)
attributes_test = attributes(spam_test, 1)
label_train = label(spam_train)
label_test = label(spam_test)


k_list = [1,5,11,21,41,61,81,101,201,401]

for k in k_list:
    print("when k = ",k,'accuracy is', accuracy(attributes_train, label_train, attributes_test,label_test, k))



# 1(b) test accuracy with z-score normalization 
    
# normalize the attributes first before classifying. 
    


avg_train = np.resize(attributes_train.mean(axis = 0),(1,attributes_train.shape[1]))
sd_train = np.resize(attributes_train.std(axis = 0),(1,attributes_train.shape[1]))
z_attributes_train = (attributes_train - avg_train)/sd_train
z_attributes_test = (attributes_test - avg_train)/sd_train


for k in k_list:
    print("when k = ",k,'accuracy is', accuracy(z_attributes_train, label_train, z_attributes_test,label_test, k))

# 1(c)


for i in range(50):
    email_id = spam_test.iloc[i,0]
    results_list= list()
    each_id_attribute = np.resize(z_attributes_test[i,],[1,57])
    for k in k_list:       
        result = KNN_classifier(z_attributes_train, label_train, each_id_attribute,k)
        for label_each in result:
            if label_each == 1:
                final_result = ("Spam")
            else:
                final_result = ("no")
        results_list.append(final_result)
    print (email_id, results_list)









