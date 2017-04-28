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
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_2 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_2.load_weights("model2.h5")
loaded_model_2.compile(loss='mean_squared_error', optimizer='adam')
print("Loaded model from disk")

# Import the data for analysis and generate realizations
fpb = np.loadtxt('SURVIVALS_Evaluate.txt')
fpb_T = fpb.T
nsamplesEQ = np.shape(fpb)[1]
nsamplesMC = 10000

## Evaluate the System Response

# Calculate the network connectivity using ANN surrogate
def CONNECTIVITY_surrogate (index):
    connect_sample_ANN = loaded_model_2.predict(fpb_T[index,:].reshape(1,-1))
    return connect_sample_ANN
start_time = time.time()
connect_ANN = Parallel(n_jobs=1)(delayed(CONNECTIVITY_surrogate)(index) for index in range(nsamplesEQ))
elapsed_time_surrogate = time.time() - start_time
print('The surrogate computational time is',elapsed_time_surrogate)
print('The mean connectivity by surrogate computation is',np.mean(connect_ANN))
print('DNN Results are now ready!')

# Calculate the network connectivity using MCS
def CONNECTIVITY_standard (index):
    connect_flag = np.zeros(nsamplesMC)
    for p in range (nsamplesMC):
        T=copy.deepcopy(G)
        for i in range (nbridges):
            if fpb_samples[p][i,index] == 1:
                if T.has_edge(int(b[i,0]),int(b[i,1])):
                    T.remove_edge(*b[i])
        if nx.has_path(T,0,11):
            connect_flag[p] = 1
    del T
    return np.mean(connect_flag)

def sample_generator(index):
    fpb_samples = np.zeros((nbridges,nsamplesEQ),dtype=int)
    for i in range (nbridges):
        for j in range (nsamplesEQ):
            if (np.random.rand() < 1. - fpb[i,j]):
                fpb_samples[i,j] = 1   # bridge will fail
    return fpb_samples

start_time = time.time()
fpb_samples = Parallel(n_jobs=24)(delayed(sample_generator)(index) for index in range(nsamplesMC))
print('MCS samples are now ready!')
connect_MCS = Parallel(n_jobs=24)(delayed(CONNECTIVITY_standard)(index) for index in range(nsamplesEQ))
elapsed_time_standard = time.time() - start_time

print('The standard computational time is',elapsed_time_standard)
print('The mean connectivity by standard computation is',np.mean(connect_MCS))
np.savetxt('connectivity_results_standard.txt',connect_MCS)
np.savetxt('connectivity_results_surrogate.txt',connect_ANN)

