# %reset
## This code builds an ANN surrogate model for two-terminal connectivity.
## Written by Mohammad Amin Nabian, mnabia2@illinois.edu, March 2017

## Import libraries
import numpy as np
import networkx as nx
import random
import copy
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn import cross_validation
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras import optimizers

## Initialization
nbridges=39

## Generating the network topology
G=nx.Graph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_node(7)
G.add_node(8)
G.add_node(9)
G.add_node(10)
G.add_node(11)
G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(1,3)
G.add_edge(1,4)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(2,6)
G.add_edge(3,4)
G.add_edge(3,5)
G.add_edge(4,7)
G.add_edge(6,7)
G.add_edge(5,6)
G.add_edge(5,8)
G.add_edge(6,9)
G.add_edge(7,8)
G.add_edge(7,11)
G.add_edge(8,10)
G.add_edge(8,9)
G.add_edge(9,10)
G.add_edge(10,11)

## Assign the failure probabilities
s=(nbridges,2);b=np.zeros(s)
b[0]=(0,1)
b[1]=(0,2)
b[2]=(0,2)
b[3]=(0,2)
b[4]=(0,2)
b[5]=(0,2)
b[6]=(0,2)
b[7]=(1,3)
b[8]=(1,4)
b[9]=(1,4)
b[10]=(2,3)
b[11]=(2,3)
b[12]=(2,3)
b[13]=(2,3)
b[14]=(2,3)
b[15]=(2,3)
b[16]=(2,6)
b[17]=(2,6)
b[18]=(2,6)
b[19]=(2,6)
b[20]=(3,4)
b[21]=(3,4)
b[22]=(3,4)
b[23]=(3,5)
b[24]=(3,5)
b[25]=(4,7)
b[26]=(5,6)
b[27]=(5,6)
b[28]=(7,8)
b[29]=(7,11)
b[30]=(7,11)
b[31]=(7,11)
b[32]=(7,11)
b[33]=(8,10)
b[34]=(8,10)
b[35]=(9,10)
b[36]=(9,10)
b[37]=(9,10)
b[38]=(9,10)

# Import the data for surrogate
fpb = np.loadtxt('SURVIVALS_Train.txt')
nsamplesEQ = np.shape(fpb)[1]
nsamplesMC = 100000

## load json and create model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_1 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_1.load_weights("model1.h5")
loaded_model_1.compile(loss='binary_crossentropy', optimizer='adam')
print("Loaded model from disk")

def CONNECTIVITY_train (index):
    connect_score = 0.
    for j in range (nsamplesMC):
        fpb_samples = np.zeros((1,nbridges),dtype=int)
        for i in range (nbridges):
            if (np.random.rand() < 1. - fpb[i,index]):
                fpb_samples[0,i] = 1   # bridge will fail
        connect_score += np.argmax(loaded_model_1.predict(fpb_samples),axis=-1)
    return connect_score/nsamplesMC
    
connect_matrix = Parallel(n_jobs=24)(delayed(CONNECTIVITY_train)(index) for index in range(nsamplesEQ))

# Reshape
fpb = fpb.T
connect_matrix = np.reshape(connect_matrix, (-1,1))
np.savetxt('connect_matrix',connect_matrix)
np.savetxt('fpb',fpb)
