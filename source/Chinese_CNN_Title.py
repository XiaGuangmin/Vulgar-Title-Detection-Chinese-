# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from models.convnets import ConvolutionalNet
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import os
import collections
#reload(sys)
#sys.setdefaultencoding('utf-8') 

main_dir = "/home/xgg/work/Chinese-disu_title-detector/"
SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30
MODEL_FILE = "/home/xgg/work/Chinese-disu_title-detector/embedding_model/Chinese_detector.h5"

def words_to_indices(inverse_vocabulary,words):
	return [inverse_vocabulary[word] for word in words]

inverse_vocabulary = collections.defaultdict(int)
if __name__ == '__main__':
	vocabulary = open(main_dir+"embedding_model/vocabulary").read().split("\n")
	counter = 0 
	for i,word in enumerate(vocabulary):
		counter = counter+1
		word_list = word.split(" ")
		inverse_vocabulary[word_list[0]] = i

	print "load vocabulary over!-->",len(inverse_vocabulary)

	pos_data = open(main_dir+"data/disu_seg.csv").read().split("\n")
	pos_data = sequence.pad_sequences([words_to_indices(inverse_vocabulary,sentence.split()) for sentence in pos_data], maxlen=SEQUENCE_LENGTH)
	print "load positive data over! -->",len(pos_data)
	nag_data = open(main_dir+"data/nomal_seg.csv").read().split("\n")
	nag_data = sequence.pad_sequences([words_to_indices(inverse_vocabulary,sentence.split()) for sentence in nag_data], maxlen=SEQUENCE_LENGTH)
	print "load nagetive data over! -->",len(nag_data)

	X = np.concatenate([pos_data,nag_data],axis=0)
	y = np.array([[1]*pos_data.shape[0] + [0]*nag_data.shape[0]],dtype=np.int32).T
	print "merge data over! -->",len(X),len(y)
	p = np.random.permutation(y.shape[0])
	X = X[p]
	y = y[p]
	print "random data over!-->",len(X),len(y)

	x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
	print "train_test_split over! -->",len(x_train),len(x_test)

	embedding_weigths = np.load(main_dir+"embedding_model/embeddings.npy")
	print "embedding_weigths over! -->",len(embedding_weigths)

	params = dict(vocabulary_size=len(vocabulary),embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH, embedding_weights=embedding_weigths)
	model = ConvolutionalNet(**params)

	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
	print "model compile over! -->",len(vocabulary)
	model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,nb_epoch=20,shuffle=True,callbacks=[EarlyStopping(monitor="val_loss",patience=2)])
	print "model fit over! -->"
	model.save_weights(MODEL_FILE)
	print "model save over! -->",len(x_train),len(x_test)