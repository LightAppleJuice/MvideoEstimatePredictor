# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:41:34 2017

@author: Aigul
"""
import os 
from sklearn import cross_validation
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^а-яА-Яa-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    return(words)


def Vector(train, remove_stopwords = False):
    print ("Cleaning and parsing review...\n") 
    traindata = [] 
    for i in range( 0, len(train['comment'])):
        traindata.append(" ".join(review_to_wordlist(train['comment'][i], remove_stopwords)))
    return traindata  


def TfIdf(X_all):

    tfv = TfidfVectorizer(min_df=3, max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1)
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)
    return X_all

df = pd.read_csv('./data/X_train.csv')
new_df = df[['reting', 'comment']]

labs = np.array([int(round(new_df['reting'][i]))-1 for i in range(len(new_df))])
labs = label_binarize(labs, classes=[0, 1, 2, 3, 4])
traindata = Vector(new_df)    

print("count tf-idf... ")
X = TfIdf(traindata)
print("test SVC model")
model = OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000))
'''
X_train = X[:10000]
X_test = X[10000:]
y_train = labs[:10000]
y_test = labs[10000:]
model.fit(X_train, y_train)
result = model.predict(X_test)#_proba(X_test)

for i, y in enumerate(y_test):
    #print (y, result[i])
    try:
        if np.nonzero(y)[0] != np.nonzero(result[i])[0]:
            print (y, result[i], new_df['comment'][10000+i])
    except:
        pass
'''
print("20 Fold CV Score. TF-IDF: ", np.mean(cross_validation.cross_val_score(model, X, labs, cv=20, scoring='roc_auc')))
