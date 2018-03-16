# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
import sys
import os
import logging
from nltk import word_tokenize
from time import time
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import chardet
main_path = "/home/xgg/work/Chinese-disu_title-detector/"
EMBEDDING_DIMENSION = 30

def preprocess_embeddings(embedding_dimension, vocabulary):
    
    embeddings = {}
    model = Word2Vec.load("../embedding_model/word2vec_gensim")
    
    weights = np.zeros((len(vocabulary), 50)) # existing vectors are 50-D

    for i, word in enumerate(vocabulary):
    	# print chardet.detect(word)
        try:
            a = word.decode("utf-8")
            weights[i] = np.array(model.wv[a])
            if (i % 1000 == 0):
                print i,weights[i]
        except:
            pass
        # print weights[i]

    pca = PCA(n_components=EMBEDDING_DIMENSION)
    weights = pca.fit_transform(weights)
    return weights

if __name__ == "__main__":
    vocabulary = open(main_path+"embedding_model/vocabulary.txt").read().split("\n")
    weights = preprocess_embeddings(EMBEDDING_DIMENSION, vocabulary)
    np.save(main_path+"embedding_model/embeddings.npy", weights)
