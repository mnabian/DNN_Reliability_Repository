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

# Perform the deep learning

fpb = np.loadtxt('fpb')
connect_matrix = np.loadtxt('connect_matrix')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(fpb, connect_matrix, test_size=0.0)
X_train = np.reshape(X_train,(-1,nbridges))
model = Sequential()
model.add(Dense(output_dim=256, input_shape=(nbridges,)))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.05))
model.add(Dense(output_dim=128))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.05))
model.add(Dense(output_dim=64))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.05))
model.add(Dense(output_dim=8))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.05))
model.add(Dense(output_dim=4))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=1))
from keras import metrics

model.compile(loss='mean_squared_error', optimizer='adam')
# Note: "accuracy" is effectively not defined for a regression problem, due to its continuous property.

model.fit(X_train, y_train, nb_epoch=300, batch_size=64)
deep_score = model.evaluate(X_test, y_test, batch_size=64)
print('The score of your regression is')
print(deep_score)

# serialize model to JSON
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")
np.savetxt('samples_train.txt',X_train)
np.savetxt('samples_test.txt',X_test)