# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:41:34 2017

@author: Aigul
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import re
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import argparse
import random

stopWords = [  'говно',
               'брак',
               'дерм',
               'отвратительн',
               'не покупайте',
               'еле фурычит',
               'пустая трата денег',
               'ненадежный',
               'перестал работать',
               'сгорел',
               'бесполезный']


def preDetector(reviews):
    result = [0] * len(reviews)
    for i, comment in enumerate(reviews):
        for word in stopWords:
            if word in comment:
                result[i] = 1

                break

    return result

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

if __name__ == '__main__':
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('-i', '--input',
                                  help='Path to feedback file',
                                  type=str,
                                  required=True)
    args = arguments_parser.parse_args()
    df = pd.read_csv(args.input)
    new_df = df[['reting', 'comment']]

    labs = np.array([int(round(new_df['reting'][i])) - 1 for i in range(len(new_df))])
    traindata = Vector(new_df)

    # traindata = traindata[:100]
    # labs = labs[:100]

    array = [i for i in range(len(labs))]
    random.shuffle(array)
    train_labels = array[1:round(0.80 * len(labs))]
    test_labels = array[round(0.80 * len(labs)):]

    testdata = [traindata[index] for index in test_labels]

    print("count tf-idf... ")
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1)

    tfv.fit([traindata[index] for index in train_labels])
    X = tfv.transform(traindata)
    print("test SVC model")

    model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr',
                      fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                      max_iter=1000)

    n_cv = 10
    # X = np.array(X)
    # shuffled_idx = list(range(X.shape[0]))
    tst_len = (X.shape[0]) // n_cv
    n = 0

    lsi = TruncatedSVD(2000)
    X_lsi = csr_matrix(lsi.fit_transform(X))
    X = X_lsi  # hstack([X, X_lsi], format='csr')

    X_train = X[train_labels]
    X_test = X[test_labels]
    y_train = labs[train_labels]
    y_test = labs[test_labels]

    model.fit(X_train, y_train)
    result = model.predict(X_test)  # _proba(X_test)

    tmp = preDetector(testdata)

    for i, y in enumerate(y_test):
        if tmp[i]:
            result[i] = 1
        if y == result[i]:
            # print (i, y, result[i], new_df['comment'][10000+i])
            n += 1
    print('Quality: {}'.format(n / X_test.shape[0]))
