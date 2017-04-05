from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import argparse
import pickle
import numpy as np
import re
import pickle

parser = argparse.ArgumentParser(description='MultiRank for object recognition improvement.')
parser.add_argument('-m', '--mapDir', dest='mapDir', default="../maps", help='Map directory')
parser.add_argument('-i', '--input', dest='input', default='../data/semantic/SingleSentences-sail.xml', help='input xml file')
parser.add_argument('-j', '--demo_input', dest='demo_input', default='../data/SingleSentence.xml', help='input xml file')
parser.add_argument('-s', '--source', dest='source', default='../data/sail_source', help='source states feature output')
parser.add_argument('-t', '--target', dest='target', default='../data/sail_target', help='target commands output')
parser.add_argument('-d', '--index', dest='index', default=None, help='shuffle index')
parser.add_argument('-n', '--lstm_size', dest='lstm_size', default=100, help='lstm size')

opts = parser.parse_args()


def loadPickle(path):
	pkl_file = open(path, 'rb')
	data = pickle.load(pkl_file)
	pkl_file.close()
	return data 

def loadData():
	sets = ['train','valid','test']
	names = ['feature' , 'outFeature']

	maxLen = 0

	data = []
	for s in sets:
		for n in names:
			d = loadPickle("%s.%s.%s.pkl"%(opts.source, n,s))
			data.append(d)
			if n == 'feature':
				maxLen = max(maxLen, max([len(x) for x in d]))
	return data, maxLen

[x_train, y_train, x_valid, y_valid, x_test, y_test], maxLen = loadData()

outDim = len(y_train[0])
inDim = len(x_train[0][0])

print outDim, inDim, maxLen	
batch_size = 32

def padSequence(data, maxlen):
	nFea = len(data[0][0])
	for i in range(len(data)):
		size = len(data[i])
		for j in range(maxlen-size):
			data[i].append([0 for x in range(nFea)])
	return np.array(data)
			
loss = "cosine_proximity"	
hiddenDim = 100

x_train = padSequence(x_train, maxLen)
x_test = padSequence(x_test, maxLen) 
x_valid = padSequence(x_valid, maxLen)  

print('Build model...')
model = Sequential()
#model.add(Embedding(max_features, 128))
model.add(LSTM(hiddenDim, input_shape=(maxLen, inDim))) #, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(outDim, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=100,
          validation_data=(x_valid, y_valid))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
classes = model.predict(x_test, batch_size=32)

def dumpPickle(path,data):
	output = open(path, 'wb')
	# Pickle dictionary using protocol 0.
	pickle.dump(data, output)
	output.close()

dumpPickle('result_%s.pkl'%(loss),classes)

model.save('model_%d_%s.h5'%(hiddenDim, loss))
