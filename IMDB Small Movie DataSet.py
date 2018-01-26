# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 04:47:03 2018

@author: Arjun
"""
from numpy import array
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from os import listdir #for loading the files from directory

def open_fun(file_name):
    #opening the file in read only mode
    f = open(file_name , 'r')
    #read the contents and then close the file
    content = f.read()
    f.close()
    return content
#clean up the contents
def doc_cleanup(doc):
    #splitting into tokens
    tokens = doc.split()
    #removal of punctuations using maketrans function
    tab = str.maketrans('', '', punctuation)#table giving out the rules
    tokens = [s.translate(tab) for s in tokens]#applying the rules specified to the tokens
    tokens = [word for word in tokens if word.isalpha()]#removing non alphabetic   
    common_words = set(stopwords.words('english'))#removing common words or stop words like the,is
    tokens = [word for word in tokens if not word in common_words]
    tokens = [word for word in tokens if len(word)>1]#removing single characters
    #print(tokens)
    return tokens
#Adding words to vocabulary
def add_to_vocab(filename,vocab):
    doc = open_fun(filename)#open the file
    tokens = doc_cleanup(doc)#clen up the document and obtain the clened up tokens.
    vocab.update(tokens)
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = open_fun(filename)
	# clean doc
	tokens = doc_cleanup(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)    
# to process all the documents
# load all docs in a directory
def process_docs(directory, vocab, trainable):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# Skip reviews from test data set
		if trainable and filename.startswith('cv9'):
			continue
		if not trainable and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines
# save list to file
def save_vocab(tokens, filename):
	data = '\n'.join(tokens)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
#specify dvocabulary file
neg_location = 'C:/Users/Arjun/Desktop/NLP/txt_sentoken/neg'
pos_location = 'C:/Users/Arjun/Desktop/NLP/txt_sentoken/pos'
vocab_filename = 'C:\\Users\\Arjun\\vocab.txt'
vocab = open_fun(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
positive_lines = process_docs(pos_location, vocab, True)
negative_lines = process_docs(neg_location, vocab, True)
#creating tokeniser
tokeniser = Tokenizer()
docs = positive_lines + negative_lines
tokeniser.fit_on_texts(docs)
#encoding the training data set
Xtrain = tokeniser.texts_to_matrix(docs, mode = 'freq')#creating encoding based on frequency of words
print(Xtrain.shape)
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
#testing data set
positive_lines = process_docs(pos_location, vocab, False)
negative_lines = process_docs(neg_location, vocab, False)
docs = positive_lines + negative_lines
#encoding testing dats set
Xtest = tokeniser.texts_to_matrix(docs, mode = 'freq')
print(Xtest.shape)
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))