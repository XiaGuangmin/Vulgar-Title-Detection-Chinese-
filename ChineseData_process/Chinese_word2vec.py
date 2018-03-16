# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
import sys
import os
import logging
from nltk import word_tokenize
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class MySentence(object):
	def __init__(self):
		self.dir = "../data/word_seg.con"
	def __iter__(self):
		with open(self.dir) as data_f:
			for line in data_f:
				sline = line.decode("gbk").strip().split()
				if len(sline) == 0:
					continue
				# tokenized_line = ''.join(word_tokenize(sline))
				# is_alpha_word_line = [word for word in tokenized_line]
				yield sline

def word2vec():
	begin = time()
	sentences = MySentence()
	model = Word2Vec(sentences,
		             size=50,
		             window=10,
		             min_count=10,
		             workers=4)
	model.save("../embedding_model/word2vec_gensim")

	model.wv.save_word2vec_format("../embedding_model/word2vec_org",
		                          "../embedding_model/vocabulary",
		                          binary=False)
	end = time()
	print "Total procesing time: %d seconds" % (end - begin)

def wordsimilarity():
	model = Word2Vec.load("../embedding_model/word2vec_gensim")
	simi=''
	try:
		simi=model.most_similar(u'性感',topn=10)
	except KeyError:
		print 'The word not in vocabulary!'

	for term in simi:
		print  '==============>>   %s,%s' %(term[0],term[1])


if __name__ == '__main__':
	
	word2vec()
	# wordsimilarity()


