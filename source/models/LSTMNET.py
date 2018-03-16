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
import numpy as np
import collections


def LSTM_Net(vocabulary_size,embedding_dimension,input_length,embedding_weights=None):
	model = Sequential()
	if embedding_weights is None:
		model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, trainable=False))
	else:
		model.add(Embedding(vocabulary_size,embedding_dimension,input_length=input_length,weights=[embedding_weights],trainable=False,dropout=0.2))
	
	model.add(LSTM(300,dropout_W=0.2,dropout_U=0.2))

	model.add(Dense(200))
	model.add(PReLU())
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(200))
	model.add(PReLU())
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model
