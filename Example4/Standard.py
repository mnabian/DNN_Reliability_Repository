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

# Import the data for analysis and generate realizations
fpb = np.loadtxt('SURVIVALS_Evaluate.txt')
fpb_T = fpb.T
nsamplesEQ = np.shape(fpb)[1]
nsamplesMC = 100000

# Calculate the network connectivity using MCS
def CONNECTIVITY_standard (index):
    connect_flag = 0.
    for p in range (nsamplesMC):
        T=copy.deepcopy(G)
        fpb_samples = np.zeros(nbridges)
        for i in range (nbridges):
            if (np.random.rand() < 1. - fpb[i,index]):
                fpb_samples[i] = 1   # bridge will fail
        for i in range (nbridges):
            if fpb_samples[i] == 1:
                if T.has_edge(int(b[i,0]),int(b[i,1])):
                    T.remove_edge(*b[i])
        if nx.has_path(T,0,11):
            connect_flag = connect_flag + 1.
    del T
    return connect_flag/nsamplesMC

start_time = time.time()
connect_MCS = Parallel(n_jobs=24)(delayed(CONNECTIVITY_standard)(index) for index in range(nsamplesEQ))
elapsed_time_standard = time.time() - start_time

print('The standard computational time is',elapsed_time_standard)
print('The mean connectivity by standard computation is',np.mean(connect_MCS))
np.savetxt('connectivity_results_standard.txt',connect_MCS)
