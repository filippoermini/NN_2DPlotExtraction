#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:13:36 2018
@author: filippo.ermini
"""
from __future__ import print_function
import json
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import os
import pickle


image_width = 50
image_height = 300
channels = 1
outClass = 3

batch_size = 128
epochs = 10

def createDataset(jsonFile,outClass,image_width,image_height):
    j = 0
    k = 0

    with open(jsonFile) as f:
        data = json.load(f)
    length = len(data["matrixList"]) * len(data["matrixList"][0]["sampleArray"])
    
    testDimension   = length // 4
    trainDimension  = length - testDimension 
    
    datasetTrainX = np.ndarray(shape=(trainDimension, image_width, image_height),dtype=np.float32)
    datasetTrainY = np.ndarray(shape=(trainDimension, outClass))
    
    datasetTestX = np.ndarray(shape=(testDimension, image_width, image_height),dtype=np.float32)
    datasetTestY = np.ndarray(shape=(testDimension, outClass))
    
    
    for imageData in data["matrixList"]:
        imageIndex = imageData["index"]
        sampleArray = imageData["sampleArray"]
        
        size = (image_width,image_height)
        image = Image.open('/Volumes/KINGSTON/Dataset_tesi/dataset_master/' + str(imageIndex) + '.png')
        image = image.convert('RGB').convert('LA')
        for i in range(0,len(sampleArray)):
            verticalBbox = sampleArray[i]['VerticalWindow']
            x  = int(verticalBbox['x'])
            y  = int(verticalBbox['y'])
            x1 = int(verticalBbox['x']) + int(verticalBbox['w'])
            y1 = int(verticalBbox['y']) + int(verticalBbox['h'])
            
            bbox = (x,y,x1,y1)
            
            verticalWindow = image.crop(bbox)
            verticalWindow = verticalWindow.resize(size)
            
            x_array = np.array(verticalWindow)
            x_array = x_array[:,:,0].reshape(image_width,image_height)
            
            
            if(j< trainDimension):
                datasetTrainX[j] = x_array
                datasetTrainY[j] = sampleArray[i]['ValueObject']
                j += 1
            else:
                datasetTestX[k] = x_array
                datasetTestY[k] = sampleArray[i]['ValueObject']
                k += 1
            
            if (j+k) % 250 == 0:
                print("%d images to array" % j)
            
            if (j+k) % 250 == 0:
                print("%d images to array" % (j+k))
            
    print("All images to array!")
    return [datasetTrainX,datasetTrainY,datasetTestX,datasetTestY]



jsonFile = '/Volumes/KINGSTON/Dataset_tesi/serialize3Line.json'
datasetDump = "DumpDataset.dat";
if(os.path.exists(datasetDump)):
    with open(datasetDump, "rb") as fp:   # Unpickling
        dataset = pickle.load(fp)
    print("load dump dataset")
else:
    dataset = createDataset(jsonFile,outClass,image_width,image_height)
    with open(datasetDump, "wb") as fp:   #Pickling
        pickle.dump(dataset, fp)
    print("dataset dumpCreated in file: %s", datasetDump )
def normalizeOutput(mat):
    [m,n] = mat.shape
    for i in range(0,m):
        vect = mat[i,:]
        max = np.max(vect)
        min = np.min(vect)
        v = (vect - min) / (max - min)
        mat[i,:] = v
    return mat
        
    
x_train = dataset[0]
x_test  = dataset[2]
y_train = dataset[1]
y_test  = dataset[3]


inputShape = (image_width, image_height)

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=inputShape))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3, activation='relu'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

 
 
k = 5
num_val_samples = len(x_train) // k
num_epochs = 100
all_mae_histories = []
 
for i in range(k):
    print('processing fold #', i)
    val_data    = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
 
    partial_train_data = np.concatenate(
     [x_train[:i * num_val_samples],
      x_train[(i + 1) * num_val_samples:]],
      axis=0)
    partial_train_targets = np.concatenate(
     [y_train[:i * num_val_samples],
      y_train[(i + 1) * num_val_samples:]],
      axis=0)
 
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
 
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
 
     #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
     #all_scores.append(val_mae)
 
 

#    
#class AccuracyHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.acc = []
#
#    def on_epoch_end(self, batch, logs={}):
#        self.acc.append(logs.get('acc'))
#
#history = AccuracyHistory()
#model = build_model()
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test),
#          callbacks=[history])
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#plt.plot(range(1, 11), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()

