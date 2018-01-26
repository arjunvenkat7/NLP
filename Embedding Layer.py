# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 03:38:53 2018

@author: Arjun
"""

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
#the documents to be used 
doc = ['Good Morning',
       'Good Day',
       'Bright Evening',
       'Terrible Day',
       'Dreadful session',
       'Pleasant session',
       'Hopeless',
       'Happy',
       'Frightened',
       'Grim'
       ]
#The next step is to specify the labels
label = [1,1,1,0,0,1,0,1,0,0]
#Encoding the documents
vocabulary_size = 50; #No of words
encoded_doc = [one_hot(d, vocabulary_size) for d in doc]
print(encoded_doc)
#Since all the docs should be of same length, padding has to be done
max_len = 5 #Maximum no of words in a document
padded_doc = pad_sequences(encoded_doc, maxlen=max_len, padding='post') #padding is done after the words
print(padded_doc) 
#defining the model to be used
model = Sequential()
#Adding the embedding Layer
model.add(Embedding(vocabulary_size, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
# fitting
model.fit(padded_doc, label, epochs=100, verbose=1)
# Evaluation of the model
loss, accuracy = model.evaluate(padded_doc, label, verbose=0)
print('Accuracy: %f' % (accuracy*100))