# -*- coding: utf-8 -*-

import csv
import cv2
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg    
import numpy as np
from sklearn.utils import shuffle
from utils import *

EPOCHS     = 12
BATCH_SIZE = 128

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

LinesUsed  = lines[1:] # exclude CSVFile Header
TotalLines = len(LinesUsed)

lines = shuffle(LinesUsed, random_state = 0)
SplitIdx = int(TotalLines * 0.8)
train_lines, valid_lines = lines[:SplitIdx], lines[SplitIdx:]

n_train = len(train_lines) * 3 * 2 # each csv gives centre, left, and right (then all flip)
n_valid = len(valid_lines) * 3 * 2

'''
Overfit test
To see if our model is capable of predicting correctly 3 images. This is a quick way to test that our model is implemented correctly
'''
#OverfitTest(lines[100], SimpleNet(), normalize=True)
#OverfitTest(lines[100], NVIDIANet()) # normalize already part of NVidia lambda layer

model = NVIDIANet()

Train_Gen = GetNextBatchFlip(train_lines, BATCH_SIZE)
Valid_Gen = GetNextBatch(valid_lines, BATCH_SIZE)

model.fit_generator(Train_Gen, 
                    steps_per_epoch  = (n_train//BATCH_SIZE),
                    epochs = EPOCHS,
                    validation_data  = Valid_Gen,
                    validation_steps = (n_valid//BATCH_SIZE))
                   
model.save('model.h5')