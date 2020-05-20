# -*- coding: utf-8 -*-
"""
Created on Thu May  7 04:14:29 2020

@author: Supernova
"""
import numpy as np
from scipy.special import expit, softmax

class RNN():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        #initializing size of the nueral 
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        #initializing the weights randomly
        self.weight1    = np.random.random((hidden_nodes, input_nodes))/1000
        self.weight2    = np.random.random((output_nodes, hidden_nodes))/1000
        self.rnn_weight = np.random.random((hidden_nodes, hidden_nodes))/1000
        #to save hidden layers
        self.hidden_layer = None
        #learning rate 
        self.learn_rate = learn_rate
        pass
    
    def forward(self, x):
        #rnn layer to save the previous hidden lyaer and use it again
        rnn_layer = np.zeros((self.rnn_weight.shape[0], 1))
        #to save all hidden layer from all the time 
        hidden_layers = dict()
        hidden_layers = {0 : rnn_layer}
        #loop for all the input data 
        for i, elm in enumerate(x):    
            
            input_layer  = elm
            hidden_layer = np.tanh(np.dot(self.weight1, input_layer) + np.dot(self.rnn_weight, rnn_layer))
            output_layer = softmax(np.dot(self.weight2, hidden_layer))
            
            rnn_layer = hidden_layer
            hidden_layers[i+1] = hidden_layer
            pass
        return output_layer, hidden_layers
    
    def backpropagation(self, input_layer, output_layer, rnn_hidden_layers):    
        #updated all the wieghts using gradient descent
        n = len(input_layer)
        #to save the derivative 
        d_weight1    = np.zeros(self.weight1.shape)
        d_rnn_wieght = np.zeros(self.rnn_weight.shape)
        
        d_weight2 = np.dot(output_layer, rnn_hidden_layers[n].T)
        
        d_output = np.dot(self.weight2.T, output_layer)
        
        for t in reversed(range(n)):
            temp = ((1 - rnn_hidden_layers[t + 1] ** 2) * d_output)
                        
            d_rnn_wieght += np.dot(temp, rnn_hidden_layers[t].T)
            
            d_weight1 += np.dot(temp, input_layer[t].T)
            
            d_h = np.dot(self.rnn_weight ,temp)
            pass
        # Clip to prevent exploding gradients.
        for d in [d_weight1, d_weight2, d_rnn_wieght]:
            np.clip(d, -1, 1, out=d)
            pass
        # Update weights and biases using gradient descent.
        self.weight1   -= self.learn_rate * d_weight1
        self.weight2   -= self.learn_rate * d_weight2
        self.rnn_weight -= self.learn_rate * d_rnn_wieght
        pass
    
    
    def fit(self, x, y, epochs):
        for e in range(epochs):
            loss     = 0
            accuracy = 0
            for elm0,elm1 in zip(x,y):
                output, hidden_layers = self.forward(elm0)
                loss -= np.log(output[elm1])
                accuracy += int(np.argmax(output) == int(elm1))
                d_L_d_y = output
                d_L_d_y[elm1] -= 1
                self.backpropagation(elm0, d_L_d_y, hidden_layers)
                pass
            L = loss/len(x)
            C = accuracy/len(x)
            if(e%500 == 0):
                print('--- Epoch %d' % (e ))
                print('Train:\tLoss %.3f | Accuracy: %.3f' % (L, C))
                pass
            pass
        pass
    def predict(self, x):
        output, hidden = self.forward(x)
        return output
        pass
    pass
pass