#%reset
## This code evaluates the performance (connectivity) of the network under extreme earthquake.
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

## load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Loaded model from disk")

## Evaluate the System Response

# Import the data for analysis and generate realizations
fpb = np.loadtxt('SURVIVALS_Evaluate.txt')
nsamplesEQ = np.shape(fpb)[1]
nsamplesMC = 1000
fpb_samples = np.zeros((nbridges,nsamplesEQ,nsamplesMC),dtype=int)
for i in range (nbridges):
    for j in range (nsamplesEQ):
        for k in range (nsamplesMC):
            if (np.random.rand() < 1. - fpb[i,j]):
                fpb_samples[i,j,k] = 1   # bridge will fail

# Calculate the network connectivity using ANN surrogate
def CONNECTIVITY_surrogate (index):
    class_result_flag = np.argmax(loaded_model.predict(fpb_samples[:,index,:].T),axis=-1)
    return class_result_flag
start_time = time.time()
class_result = Parallel(n_jobs=24)(delayed(CONNECTIVITY_surrogate)(index) for index in range(nsamplesEQ))
# Can't be done in parallel since keras uses all CPUs by default.
elapsed_time_surrogate = time.time() - start_time
print('The surrogate computational time is',elapsed_time_surrogate)
print('The mean connectivity by surrogate computation is',np.mean(class_result))
np.savetxt('connectivity_results_surrogate.txt',class_result)
del class_result

# Calculate the network connectivity using MCS
def CONNECTIVITY_standard (index):
    connect_flag = np.zeros(nsamplesMC)
    for p in range (nsamplesMC):
        T=copy.deepcopy(G)
        for i in range (nbridges):
            if fpb_samples[i,index,p] == 1:
                if T.has_edge(int(b[i,0]),int(b[i,1])):
                    T.remove_edge(*b[i])
        if nx.has_path(T,0,11):
            connect_flag[p] = 1
    del T
    return connect_flag

start_time = time.time()
connect_matrix = Parallel(n_jobs=24)(delayed(CONNECTIVITY_standard)(index) for index in range(nsamplesEQ))
elapsed_time_standard = time.time() - start_time
print('The standard computational time is',elapsed_time_standard)
print('The mean connectivity by standard computation is',np.mean(connect_matrix))
np.savetxt('connectivity_results_standard.txt',connect_matrix)

# Reshape
fpb_samples_reshaped = np.zeros((nsamplesEQ*nsamplesMC,nbridges),dtype=int)
for i in range (nbridges):
    c = -1
    for j in range (nsamplesEQ):
        for k in range (nsamplesMC):
            c += 1
            fpb_samples_reshaped[c,i] = fpb_samples[i,j,k]
            
np.savetxt('samples_evaluate.txt',fpb_samples_reshaped)