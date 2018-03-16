# coding: utf-8

import sys
import jieba
import importlib
import chardet  
from gensim.models import Word2Vec

# reload(sys)
# sys.setdefaultencoding("utf-8") 


disu_f = open("../data/disu.csv")
nomal_f = open("../data/nomal.csv")

# word_seg_f = open("../data/word_seg.con","w")
word_seg_disu_f = open("../data/disu_seg.csv","w")
word_seg_nomal_f = open("../data/nomal_seg.csv","w")
list_disu = []
list_nomal = []
counter = 0
for sentence in disu_f:
#	print type(sentence),chardet.detect(sentence)
	temp_word = jieba.cut(sentence)
	list_disu.append(" ".join(temp_word))
	counter = counter + 1
	if(counter % 1000 == 0):
		print "line counter: " ,counter,len(list_disu)

for seg in list_disu:
	out_list = [seg.replace(","," ").strip().encode("utf-8"),"1"]
	print ",".join(out_list)
	word_seg_disu_f.write(",".join(out_list))
	word_seg_disu_f.write("\n")
word_seg_disu_f.close()

for sentence in nomal_f:
	temp_word = jieba.cut(sentence)
	list_nomal.append(" ".join(temp_word))
	counter = counter + 1
	if(counter % 1000 == 0):
		print "line counter: " ,counter,len(list_nomal)

for seg in list_nomal:
	out_list = ["0",seg.replace(","," ").strip().encode("utf-8")]
	# print ",".join(out_list)
	word_seg_nomal_f.write(",".join(out_list))
	word_seg_nomal_f.write("\n")
word_seg_nomal_f.close()
	


