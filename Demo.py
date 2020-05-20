# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:39:10 2020

@author: Supernova
"""
import numpy as np
from data import train_data, test_data
from RNN import RNN

# la construction de notre dictionnaire 
# c'est notre wordnet
vocabulaire = [w for text in train_data.keys() for w in text.split(' ')]
vocabulaire = list(set(vocabulaire))
print(vocabulaire)

print("Nous avons ", len(vocabulaire)," mot unique ")

vocabulaire_test = [w for text in test_data.keys() for w in text.split(' ')]
vocabulaire_test = list(set(vocabulaire))

print("Nous avons ", len(vocabulaire_test)," mot unique pour le test ")

word_to_id = { w: i for i, w in enumerate(vocabulaire) }
id_to_word = { i: w for i, w in enumerate(vocabulaire) }

"""///////////////////////////////////////////////////////////////////////////"""
# word into vector 
def text_into_array(text):
  inputs = list()
  for w in text.split(' '):
    v = np.zeros((len(vocabulaire), 1))
    v[word_to_id[w]] = 1
    inputs.append(v)
  return inputs



"""///////////////////////////////////////////////////////////////////////////"""
x_train = []
y_train = []
for key,value in list(train_data.items()):
    x_train.append(text_into_array(key))
    y_train.append(int(value))
    pass
pass


RNN = RNN(18, 42, 2, 0.1)
RNN.fit(x_train, y_train, 2000)
t = text_into_array('right')
print(RNN.predict(t))