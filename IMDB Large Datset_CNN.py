# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 01:27:55 2018

@author: Arjun
"""
# IMDB large data set review
# I have used the following neural network structure
# Embedding layer followed by a drop out, then a 1D convolution, then a hidden layer and then the output layer
# The accuracy which I obtained is around 89% Still there is scope for improvement

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D
#define the top words, in our case we can take the first 5000 words
top_features = 5000
#define the max length of the review to be considered(no of words)t 
max_length = 500
#The dimensions for the word vector 
dimensions = 32
# FIlter size and kernels 
filters = 64;
kernels = 3;
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = top_features)
#Restrict or incerase the review size based on max_length
X_train = sequence.pad_sequences(X_train, maxlen = max_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_length)
#now defining our model
model = Sequential()
#Our sturcture is embedding layer, followed by hidden layer and then output
model.add(Embedding(top_features,dimensions,input_length = max_length))
#Adding drop out
model.add(Dropout(0.3))
#adding a 1d convolution layer
model.add(Conv1D(filters,kernels,padding='valid',activation='relu',strides=1))
#Max pooling
model.add(MaxPooling1D(pool_size = 2))
#Flatten 
model.add(Flatten())
#hidden layer
model.add(Dense(250, activation = 'relu'))
model.add(Dropout(0.4))
#output layer
model.add(Dense(1, activation = 'sigmoid'))
#compiling and checking for accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
print(model.summary())
#Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
