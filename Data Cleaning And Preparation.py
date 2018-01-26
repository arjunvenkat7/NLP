# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 04:44:38 2018

@author: Arjun
"""
#IMDB movie review analysis
#Data Preparation Part
#Loading the data
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
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
# to process all the documents
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_to_vocab(path, vocab)
    # save list to file
def save_vocab(tokens, filename):
	data = '\n'.join(tokens)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
# specify directory to load
vocab = Counter()
neg_location = 'C:/Users/Arjun/Desktop/NLP/txt_sentoken/neg'
pos_location = 'C:/Users/Arjun/Desktop/NLP/txt_sentoken/pos'
# add all docs to vocab
process_docs(neg_location, vocab)
process_docs(pos_location, vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with > 5 occurrence
min_occurence = 5
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print(len(tokens))
# save tokens to a vocabulary file
save_vocab(tokens, 'vocab.txt')
    