# coding: utf-8

from models.convnets import ConvolutionalNet
from keras.models import load_model
from keras.preprocessing import sequence
#from preprocessors.preprocess_text import clean
from models.LSTMNET import LSTM_Net
import numpy as np
import sys
import string 
import re
import jieba
import collections
import chardet

reload(sys)
sys.setdefaultencoding('utf-8')

SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30
main_dir = "/home/xgg/work/Chinese-disu_title-detector/"
model_lstm_path = main_dir+'embedding_model/lstm_weights.h5'
model_cnn_path = main_dir+'embedding_model/Chinese_detector.h5'
embedding_weights = np.load(main_dir+"embedding_model/embeddings.npy")

inverse_vocabulary = collections.defaultdict(int)
vocabulary = open(main_dir+'embedding_model/vocabulary.con','r').read().split("\n")
# for k in vocabulary:
	# print chardet.detect(k)
for i,word in enumerate(vocabulary):
	if len(word) > 0:
		inverse_vocabulary[word] = i

def words_to_indices(inverse_vocabulary,words):
	return [inverse_vocabulary[word] for word in words]

class Predictor(object):
	"""docstring for Predictor"""
	def __init__(self, model_style):
		super(Predictor, self).__init__()
		self.model_style = model_style
		if(self.model_style == 'CNN' or self.model_style == 'cnn'):
			self.model_path = model_cnn_path
			model = ConvolutionalNet(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH)
		if (self.model_style == "LSTM" or self.model_style == "lstm"):
			self.model_path = model_lstm_path
			params = dict(vocabulary_size=len(vocabulary),embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH,embedding_weights=embedding_weights)
			model =  LSTM_Net(**params)
		model.load_weights(self.model_path)
		self.model = model

	def predict(self,headline):
		try:
			headline = headline
		except:
			headline = headline.decode('gbk').encode('utf-8')
		#word_seg = list(jieba.cut(headline))
		word_seg = [_.encode('utf-8') for _ in list(jieba.cut(headline))]
		#a = word_seg[0]
		#print "===========>",word_seg[0],chardet.detect(a)
		inputs = sequence.pad_sequences([words_to_indices(inverse_vocabulary,word_seg)],maxlen=SEQUENCE_LENGTH)
		clickbaitiness = self.model.predict(inputs)[0,0]
		return clickbaitiness

if __name__ == "__main__":
	predictor = Predictor(sys.argv[1]) 
	print(sys.argv[2])
	#predictor.predict(sys.argv[2])
	print ("headline is {0} % disu_title".format(round(predictor.predict(sys.argv[2]) * 100, 2)))	