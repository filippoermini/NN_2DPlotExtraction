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

image_width = 50
image_height = 300
channels = 4
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
    
    datasetTrainX = np.ndarray(shape=(trainDimension, image_width, image_height,channels),dtype=np.float32)
    datasetTrainY = np.ndarray(shape=(trainDimension, outClass))
    
    datasetTestX = np.ndarray(shape=(testDimension, image_width, image_height,channels),dtype=np.float32)
    datasetTestY = np.ndarray(shape=(testDimension, outClass))
    
    
    for imageData in data["matrixList"]:
        imageIndex = imageData["index"]
        sampleArray = imageData["sampleArray"]
        
        size = (image_width,image_height)
        image = Image.open('/Volumes/KINGSTON/Dataset_tesi/dataset_master/' + str(imageIndex) + '.png')
        for i in range(0,len(sampleArray)):
            verticalBbox = sampleArray[i]['VerticalWindow']
            x  = int(verticalBbox['x'])
            y  = int(verticalBbox['y'])
            x1 = int(verticalBbox['x']) + int(verticalBbox['w'])
            y1 = int(verticalBbox['y']) + int(verticalBbox['h'])
            
            bbox = (x,y,x1,y1)
            
            verticalWindow = image.crop(bbox)
            verticalWindow = verticalWindow.resize(size)
            
            x_array = img_to_array(verticalWindow)
            x_array = x_array.reshape(image_width,image_height,channels)
            
            
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
            
    print("All images to array!")
    return [datasetTrainX,datasetTrainY,datasetTestX,datasetTestY]

jsonFile = '/Volumes/KINGSTON/Dataset_tesi/serialize3Line.json'
dataset = createDataset(jsonFile,outClass,image_width,image_height)


x_train = dataset[0]
x_test  = dataset[2]
y_train = dataset[1]
y_test  = dataset[3]

input_shape = (image_width, image_height, 4)

model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(image_width, image_height, 4)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


#model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()