# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:35:15 2018

@author: Arjun
"""

from gensim.models import Word2Vec
# Load the training data
filename = 'C:/Users/Arjun/Desktop/NLP/book.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
#perform tokenisation on the data
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
n = 3;
for i in range(len(tokens)):
  output =  [tokens[i:i+n] for i in range(0, len(tokens), n)]
# train model
model = Word2Vec(output, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['EBook'])

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
# train model
model = Word2Vec(output, min_count=1)
# fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()