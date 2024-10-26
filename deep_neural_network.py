import sys
import numpy as np
import pandas as pd
import os
import time
import keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasClassifier
from keras import backend as K
import tensorflow as tf
from keras.initializers import Initializer
 
seed = 7
np.random.seed(seed)

from tensorflow.keras.utils import to_categorical

from google.colab import files 
dataset =  pd.read_csv('/content/drive/MyDrive/Colab Notebooks/protein_sec/dataset_sec_protein.csv')

#print(dataset)
#Spliting data into training and testing
X = dataset.iloc[:,1:261].values
#print(X)
Y = dataset.iloc[:,261].values
print(Y)

#Label the class

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(Y)
#print(y1)
Y = pd.get_dummies(y1).values
print(Y)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD,Adam

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def baseline_model():

  model = Sequential()
  model.add(Dense(128,input_shape=(260,),activation='relu', kernel_initializer='he_normal'))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(3,activation='softmax'))
  model.compile(Adam(lr=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])
  model.summary()
  return model

#Fitting the model and predicting

start = time.time()
batch_size = 8
print ("Using batchsize = {} ".format (batch_size))

estimator = KerasClassifier(build_fn=baseline_model, epochs=50,batch_size=batch_size)
kfold  = KFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
 
print("Average Acc: %.2f%% and StDev: (%.2f%%)" % (results.mean()*100, results.std()*100))
  
end = time.time()

print("Training Time", end-start)
