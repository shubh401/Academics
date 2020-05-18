from __future__ import unicode_literals

import re
import emoji
import torch
import pickle
import csv
import nltk
import numpy as np
import pandas as pd
from network import *
from sklearn import svm
from dataloader import *
from twokenize import *
from nltk import tokenize
from torch.utils import data
from collections import Counter
from sklearn import decomposition
from twokenize import simpleTokenize
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def readtsv(file):
    data = []  
    """0 is not offensive, 1 is offensive"""
    with open(file, 'r') as file:
        my_reader = csv.reader(file, delimiter='\t')
        for row in my_reader:
            if 'OFF' in row[0]:
                data.append([1, row[1]])
            else:
                data.append([0, row[1]])
    return data

def readtest(file):
    data = []  
    """0 is not offensive, 1 is offensive"""
    with open(file, 'r') as file:
        my_reader = csv.reader(file, delimiter='\t')
        for row in my_reader:
            data.append(row)
    return data

def featureextract(train_data, dev_data, test_data):

    """Standard features and labels"""
    features_data_train = []
    labels_train = []

    for item in train_data:
        features = []
        itemstr = item[1]
        itemsent = nltk.tokenize.sent_tokenize(itemstr)
        """Length of Sentences"""
        features.append(len(itemsent))
        """Mean Words per Sentences"""
        features.append(np.asarray([len(item.split()) for item in
                        itemsent]).mean())  
        features_data_train.append(features)
        labels_train.append(item[0])

    features_data_train = np.asarray(features_data_train)
    labels_train = np.asarray(labels_train).reshape(-1, 1)

    features_data_dev = []
    labels_dev = []

    for item in dev_data:
        features = []
        itemstr = item[1]
        itemsent = nltk.tokenize.sent_tokenize(itemstr)
        """Length of Sentences"""
        features.append(len(itemsent))
        """Mean Words per Sentences"""
        features.append(np.asarray([len(item.split()) for item in
                        itemsent]).mean())  
        features_data_dev.append(features)
        labels_dev.append(item[0])

    features_data_dev = np.asarray(features_data_dev)
    labels_dev = np.asarray(labels_dev).reshape(-1, 1)

    features_data_test = []

    for item in test_data:
        features = []
        itemstr = item[0]
        itemsent = tokenize.sent_tokenize(itemstr)
        """Length of Sentences"""
        features.append(len(itemsent))
        """Mean Words per Sentences"""
        features.append(np.asarray([len(item.split()) for item in
                        itemsent]).mean())  
        features_data_test.append(features)

    features_data_test = np.asarray(features_data_test)

    """TFID Vectorizer"""
    tweets_train = [item[1] for item in train_data]
    tweets_dev = [item[1] for item in dev_data]

    vectorizer = TfidfVectorizer(
        tokenizer = simpleTokenize,
        analyzer = 'char_wb',
        ngram_range = (3, 6),
        stop_words = nltk.corpus.stopwords.words('english'),
        use_idf = True,
        smooth_idf = False,
        norm = None,
        decode_error = 'replace',
        max_features=10000,
        min_df = 5,
        max_df = 0.75,
        )

    tfidf_train = vectorizer.fit_transform(tweets_train).toarray()
    tfidf_dev = vectorizer.transform(tweets_dev).toarray()
    tfidf_test = vectorizer.transform([item[0] for item in
            test_data]).toarray()

    train_data = np.concatenate((features_data_train, tfidf_train), 1)
    dev_data = np.concatenate((features_data_dev, tfidf_dev), 1)
    test_data = np.concatenate((features_data_test, tfidf_test), 1)

    train_data = np.concatenate((labels_train, train_data), 1)
    dev_data = np.concatenate((labels_dev, dev_data), 1)

    return (train_data, dev_data, test_data)

def SVMClassifier(train_data, dev_data, test_data):
    clf = svm.SVC(gamma='scale', kernel='linear', C=1.0, verbose=True)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    pred = clf.predict(dev_data[:, 1:])
    vlabel = dev_data[:, 0]
    pred_test = clf.predict(test_data)
    return (vlabel, pred, pred_test)

def NaiveBayesClassifier(train_data, dev_data, test_data):
    model = GaussianNB().fit(train_data[:, 1:], train_data[:, 0])
    pred = model.predict(dev_data[:, 1:])
    vlabel = dev_data[:, 0]
    pred_test = model.predict(test_data)
    return (vlabel, pred, pred_test)

def NeuralNetwork(train_data,dev_data,test_data):
    D_in,H,D_out=1000,100,1
    
    """Parameters"""
    params = {'batch_size': 64,
              'shuffle': True,
              }
    model = EyeNet(D_in,H,D_out)    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()    
    max_epochs = 100
    
    us_train_data_pos=[]
    us_train_data_neg=[]
    
    for i in range(train_data.shape[0]):
        if train_data[i,0]==0 and len(us_train_data_neg)<=3399:
            us_train_data_neg.append(train_data[i,:])
            
        if train_data[i,0]==1 and len(us_train_data_pos)<=3399:
            us_train_data_pos.append(train_data[i,:])
            
    train_data = np.asarray(us_train_data_neg+us_train_data_pos)   

    pca = PCA(n_components=1000)
    pca.fit_transform(train_data[:,1:])
    train_data=np.concatenate((train_data[:,0].reshape(-1,1),pca.transform(train_data[:,1:])),1) 
    dev_data=np.concatenate((dev_data[:,0].reshape(-1,1),pca.transform(dev_data[:,1:])),1)     
            
    """Generators"""    
    training_set = eyeDataset(train_data)
    training_generator = data.DataLoader(training_set, **params)
    
    validation_set = eyeDataset(dev_data)
    validation_generator = data.DataLoader(validation_set, **params)
    
    for epoch in range(max_epochs):
        """Training"""
        loss_acc = []
        for local_batch, local_labels in training_generator:
            y_pred = model(local_batch.float())
            loss = criterion(y_pred.flatten(), local_labels.float())    
            loss_acc.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for local_batch, local_labels in validation_generator:
            y_pred = model(local_batch.float())
            loss_val = criterion(y_pred.flatten(), local_labels.float()) 
            
        if (epoch % 10 == 0):    
            print("\nEpoch %d Training Loss"%epoch,np.mean(np.asarray(loss_acc)))
        if (epoch % 10 == 0):    
            print("\nEpoch %d Validation Loss"%epoch,loss_val.item())
    gt = dev_data[:,0]
    
    pred_test = (np.round(model(torch.from_numpy(pca.transform(test_data)).float()).data.numpy()))
    pred = (np.round(model(torch.from_numpy(dev_data[:,1:]).float()).data.numpy()))
    
    pred_test_list = []
    pred_list = []
    for element in pred_test:
        pred_test_list.extend(element.tolist())
    for element in pred:
        pred_list.extend(element.tolist())
         
    return gt, pred_list, pred_test_list


def accuracy_test(gt, pred):
    conf_mat = confusion_matrix(gt, pred)
    return (np.sum(gt == pred) / gt.shape[0] * 100, conf_mat)


"""READING & PREPROCESSING FILES USING TWOKENIZE.PY"""
"""EXTRA PREPROCESSING STEP TO SPLIT EMOJIS ADDED IN TWOKENIZE.PY"""                     
train_data  =readtsv("train.tsv")
for rows in train_data:
    rows[1] = tokenizeText(rows[1])
test_data = readtest("test.tsv")
for rows in test_data:
    rows[0] = tokenizeText(rows[0])
dev_data = readtsv("dev.tsv")
for rows in dev_data:
    rows[1] = tokenizeText(rows[1])

"""SURFACE LEVEL FEATURE EXTRACTION"""
train_data,dev_data,test_data = featureextract(train_data,dev_data,test_data)

"""NAIVE BAYES CLASSIFIER"""
gtnb, prednb, pred_testnb = NaiveBayesClassifier(train_data, dev_data, test_data)
print("\nNaive Bayes DEV CLASSIFICATION ACCURACY IS: ",accuracy_test(gtnb, prednb)[0])
print (accuracy_test(gtnb, prednb)[1])

print("\nNaive Bayes TEST CLASSIFICATION ACCURACY IS: ",accuracy_test(gtnb,pred_testnb)[0])
print (accuracy_test(gtnb, pred_testnb)[1])

"""GENERATING PREDICTION.TXT FOR TEST.TSV"""
with open('prediction1.txt', 'w') as f:
    for prediction in pred_testnb.tolist():
        if(int(prediction) == 0):
            f.writelines("OFF\n")
        else:
            f.writelines("NOT\n")

"""SVM CLASSIFIER - Linear Kernel"""
gtsvm, predsvm, predtestsvm = SVMClassifier(train_data, dev_data, test_data)
print("\nSVM TEST CLASSIFICATION ACCURACY IS: ",accuracy_test(gtsvm, predtestsvm)[0])
print (accuracy_test(gtsvm, predsvm)[1])

print("\nSVM DEV CLASSIFICATION ACCURACY IS: ",accuracy_test(gtsvm, predsvm)[0])
print (accuracy_test(gtsvm, predsvm)[1])

"""NEURAL NETWORK APPROACH"""
gtnn, prednn, pred_testnn = NeuralNetwork(train_data, dev_data, test_data)
print("\nNeural Network DEV CLASSIFICATION ACCURACY IS: ",accuracy_test(gtnn,prednn)[0])
print (accuracy_test(gtnn, prednn)[1])
"""GENERATING PREDICTION.TXT FOR TEST.TSV"""
with open('prediction.txt', 'w') as f:
    for prediction in prednn:
        if(int(prediction) == 0):
            f.writelines("OFF\n")
        else:
            f.writelines("NOT\n")

print("\nNeural Network TEST CLASSIFICATION ACCURACY IS: ",accuracy_test(gtnn,prednn)[0])
print (accuracy_test(gtnn, pred_testnn)[1])