#coding: utf-8

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence,text
from sklearn.model_selection import train_test_split
from models.LSTMNET import LSTM_Net
import numpy as np
import collections

main_dir = "/home/xgg/work/Chinese-disu_title-detector/"
EMBEDDING_DIMENSION=30
SEQUENCE_LENGTH = 20
counter = 0

inverse_vocabulary = collections.defaultdict(int)
vocabulary = open(main_dir+"embedding_model/vocabulary").read().split("\n")
embedding_weights = np.load(main_dir+"embedding_model/embeddings.npy")

def words_to_indices(inverse_vocabulary,words):
	return [inverse_vocabulary[word] for word in words]

for i,word in enumerate(vocabulary):
	counter = counter+1
	word_list = word.split(" ")
	inverse_vocabulary[word_list[0]] = i

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

params = dict(vocabulary_size=len(vocabulary),embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH,embedding_weights=embedding_weights)
model = LSTM_Net(**params)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint=ModelCheckpoint(main_dir+"embedding_model/lstm_weights.h5",monitor='val_acc',save_best_only=True,verbose=2)
model.fit(x_train,y_train,validation_data=(x_test,y_test),
	      batch_size=64,nb_epoch=50,verbose=2,validation_split=0.1,
	      shuffle=True,callbacks=[checkpoint])