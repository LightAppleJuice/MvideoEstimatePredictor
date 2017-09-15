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
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from bs4 import BeautifulSoup
import re
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, vstack, hstack


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = review_text.replace(')', ' ) ')
    review_text = review_text.replace('(', ' ( ')
    review_text = review_text.replace('!', ' ! ')
    review_text = review_text.replace('?', ' ? ')
    review_text = review_text.replace('+', ' + ')
    review_text = re.sub("[^а-яА-Яa-zA-Z()\-+!?0-9]", " ", review_text)
    words = review_text.lower().split()
    return (words)


# def review_to_wordlist( review, remove_stopwords=False ):
#     review_text = BeautifulSoup(review).get_text()
#     review_text = re.sub("[^а-яА-Яa-zA-Z]"," ", review_text)
#     words = review_text.lower().split()
#     return(words)

def Vector(train, remove_stopwords=False):
    print("Cleaning and parsing review...\n")
    traindata = []
    for i in range(0, len(train['comment'])):
        traindata.append(" ".join(review_to_wordlist(train['comment'][i], remove_stopwords)))
    return traindata


def TfIdf(X_all):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfv.fit(X_all)
    X_all = tfv.transform(X_all)
    return X_all


df = pd.read_csv('X_train.csv')
new_df = df[['reting', 'comment']]

labs = np.array([int(round(new_df['reting'][i])) - 1 for i in range(len(new_df))])
# labs = label_binarize(labs, classes=[0, 1, 2, 3, 4])
traindata = Vector(new_df)

# labs= labs[:100]
# traindata = traindata[:100]
print("count tf-idf... ")
# X = TfIdf(traindata)
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1)

tfv.fit(traindata[:round(0.60 * len(labs))])
# tfv.fit(traindata)

X = tfv.transform(traindata)
print("test SVC model")

# print ("count bag of words")
# count = CountVectorizer()
# X = count.fit_transform(traindata)
# print ("test SVC model")


model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr',
                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                  max_iter=1000)

n_cv = 10
# X = np.array(X)
shuffled_idx = list(range(X.shape[0]))
tst_len = (X.shape[0]) // n_cv
n = 0

lsi = TruncatedSVD(2000)
X_lsi = csr_matrix(lsi.fit_transform(X))
X = X_lsi  # hstack([X, X_lsi], format='csr')

X_train = X[:round(0.60 * len(labs))]
X_test = X[round(0.80 * len(labs)):]
y_train = labs[:round(0.60 * len(labs))]
y_test = labs[round(0.80 * len(labs)):]
model.fit(X_train, y_train)
result = model.predict(X_test)  # _proba(X_test)

for i, y in enumerate(y_test):
    if y == result[i]:
        # print (i, y, result[i], new_df['comment'][10000+i])
        n += 1

        # for i in range(n_cv):
        #     train_idx = shuffled_idx[:i*tst_len] + shuffled_idx[(i+1)*tst_len:]
        #     train_feats = X[train_idx]
        #     train_labs = labs[train_idx]
        #     model.fit(train_feats, train_labs)
        #
        #     test_idx = shuffled_idx[i*tst_len:(i+1)*tst_len]
        #     test_feats = X[test_idx]
        #     test_labs = labs[test_idx]
        #
        #     model.fit(train_feats, train_labs)
        #     result = model.predict(test_feats)
        #     for i, y in enumerate(test_labs):
        #         if y == result[i]:
        #             print (i, y, result[i], new_df['comment'][10000+i])
        # n+=1
print(n / X_test.shape[0])
# print ("10 Fold CV Score. TF-IDF: ", np.mean(cross_validation.cross_val_score(model, X, labs, cv=10, scoring='accuracy')))
