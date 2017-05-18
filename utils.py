# -*- coding: utf-8 -*-

import csv
import cv2
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg    
import numpy as np
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import regularizers

CORRECTION = 0.2
L2NORM     = 0.0008
ACTIVATION = 'elu'

def Get3ImagesAndLabels(line):  
    '''Return image/measurement of centre, left, right of a particular CSV entry'''
    
    kCamera = 3        
    images = []    
    measurements = []            
    for camera in range(kCamera):
        source_path = line[camera]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = mpimg.imread(current_path)
        images.append(image)
        measurement = float(line[3])                
        if camera is 1:   # left camera
            measurement = measurement + CORRECTION
        elif camera is 2: # right camera
            measurement = measurement - CORRECTION
        measurements.append(measurement)
        
    return np.array(images), np.array(measurements)


def OverfitTest(line, model, normalize=False):
    '''Repeatly train the same centre, left, right image as reasonability test to see if our model can overfit'''
    
    X_train1, y_train1 = Get3ImagesAndLabels(line)
    if normalize:
        X_train1       = X_train1/255.0-0.5        
    model.fit(X_train1, y_train1, batch_size=3, epochs=1000, verbose=0)
    prediction = np.transpose(model.predict(X_train1, batch_size=3).squeeze())    
    print('Predicted     [{0:10.7f} {1:10.7f} {2:10.7f}]'.format(*prediction[0:3]))
    print('Actual Labels [{0:10.7f} {1:10.7f} {2:10.7f}]'.format(*y_train1[0:3]))


def GetNextBatch(CSVData, batch_size):
    '''Generator to read CSV lines to produce batch of centre, left, right images
    In other words, if batch_size is 128, only 128/3 + 1 CSV lines are read in each step
    '''
    
    kData  = len(CSVData)
    offset = 0    
    kCamera = 3
    batch_size_CSV = 128//kCamera
    while True:
        images = []    
        measurements = []
        offset = (batch_size_CSV + offset) % (kData - batch_size_CSV)        
        for i in range(batch_size_CSV + 1):
            line = CSVData[offset+i]
            for camera in range(kCamera):
                source_path = line[camera]
                filename = source_path.split('/')[-1]
                current_path = './data/IMG/' + filename # includes both Udacity sample and our recorded recovery images
                image = mpimg.imread(current_path)
                images.append(image)
                measurement = float(line[3])                
                if camera is 1:   # left camera
                    measurement = measurement + CORRECTION
                elif camera is 2: # right camera
                    measurement = measurement - CORRECTION
                measurements.append(measurement)
        
        images = np.array(images[:batch_size])
        yield images, np.array(measurements[:batch_size])

def GetNextBatchFlip(CSVData, batch_size):
    '''Generator to read CSV lines to produce batch of centre, left, right, flipped images
    In other words, if batch_size is 128, only 128/6 + 1 CSV lines are read in each step
    '''
    
    kData  = len(CSVData)
    offset = 0    
    kCamera = 3    
    batch_size_flip = int(batch_size/2)    
    batch_size_CSV  = batch_size_flip//kCamera    
    while True:
        images = []    
        measurements = []
        offset = (batch_size_CSV + offset) % (kData - batch_size_CSV)        
        for i in range(batch_size_CSV + 1):
            line = CSVData[offset+i]
            for camera in range(kCamera):
                source_path = line[camera]
                filename = source_path.split('/')[-1]
                current_path = './data/IMG/' + filename
                image = mpimg.imread(current_path)
                images.append(image)
                measurement = float(line[3])                
                if camera is 1:   # left camera
                    measurement = measurement + CORRECTION
                elif camera is 2: # right camera
                    measurement = measurement - CORRECTION
                measurements.append(measurement)
        
        augmented_images = []
        augmented_measurements = []
        for image, measurement in zip(images, measurements):
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            flipped_image = cv2.flip(image, 1)
            flipped_measurement = measurement * -1.0
            augmented_images.append(flipped_image)
            augmented_measurements.append(flipped_measurement)
                
        augmented_images = np.array(augmented_images[:batch_size])        

        yield augmented_images, np.array(augmented_measurements[:batch_size])


def SimpleNet():
    '''1 Fully connected network to test pipeline'''
    
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss ='mse')
    return model


def NVIDIANet():    
    '''This is a modified NVidia framework. The main changes we made were:
       1. input size from 66x200x3 to 160x320x
       2. same padding for all convolutional layers instead of valid padding
       3. stride of (2,2) for all convolutional layers instead of only the first three
       4. additional cropping layer of top 60 and lower 20 pixels
       
       References:
       Bojarski, Testa, et al - End to End Learning for Self-Driving Cars
       https://arxiv.org/abs/1604.07316
    '''    
    
    model = Sequential()
    
    '''Preprocessing layers'''    
    # Layer 1, 2 - Normalization and Cropping
    # Input :  160x320x3
    # Output:   80x320x3
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))    
    model.add(Cropping2D(cropping=((60,20), (0,0))))

    '''Convolutional layers'''
    # Layer 3,4,5 - Convolution, elu activation, and maxpooling
    # input : 80x320x 3
    # output: 39x159x24
    # Parameters: 1824 = 3*5*5*24 + 24
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(L2NORM)))
    model.add(Activation(ACTIVATION))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    # Layer 6, 7, 8, 9 - Convolution, elu activation, maxpooling, and dropout
    # input : 39x159x24
    # output: 19x 79x36
    # Parameters: 21636 = 24*5*5*36 + 36
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(L2NORM)))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Dropout(0.5))

    # Layer 10, 11, 12 - Convolution, elu activation, and maxpooling
    # input : 19x79x36
    # output:  9x39x48
    # Parameters: 43248 = 36*5*5*48 + 48
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(L2NORM)))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # Layer 13, 14, 15 - Convolution, elu activation, and maxpooling
    # input :  9x39x48
    # output:  4x19x64
    # Parameters: 27712 = 48*3*3*64 + 64
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(L2NORM)))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # Layer 16, 17, 18 - Convolution, elu activation, and maxpooling
    # input :  4x19x64
    # output:  1x 9x64
    # Parameters: 36928 = 64*3*3*64 + 64
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(L2NORM)))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    '''Fully connected layers'''
    # Layer 19, 20 - Fully connected, and elu activation
    # input :  1x9x64 = 576
    # output:  1164
    # Parameters: 671628 = 576*1164 + 1164
    model.add(Flatten())    
    model.add(Dense(1164))
    model.add(Activation(ACTIVATION))

    # Layer 21, 22 - Fully connected, and elu activation
    # input :  1164
    # output:  100
    # Parameters: 116500 = 1164*100 + 100
    model.add(Dense(100))
    model.add(Activation(ACTIVATION))

    # Layer 23, 24 - Fully connected, and elu activation
    # input :  100
    # output:   50
    # Parameters: 5050 = 100*50 + 50
    model.add(Dense(50))
    model.add(Activation(ACTIVATION))

    # Layer 25, 26 - Fully connected, and elu activation
    # input :  50
    # output:  10
    # Parameters: 510 = 50*10 + 10
    model.add(Dense(10))
    model.add(Activation(ACTIVATION))

    # Layer 27 - Fully connected
    # input :  10
    # output:   1
    # Parameters: 11 = 10*1 + 1
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer='adam', loss='mse')
    
    return model
