import sys

def readData(filename):
	data = []
	for line in open(filename):
		line = line.replace("<s>","")
		data.append(line.strip().lower())
	return data

ref = readData(sys.argv[1])
res = readData(sys.argv[2])

import nltk
import numpy as np 

scores = []
#the maximum is bigram, so assign the weight into 2 half.
for i in range(len(ref)):
	scores.append(nltk.translate.bleu_score.sentence_bleu([ref[i]], res[i]))
print np.mean(scores)
