import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import keras
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

import pandas as pd
from pandas import Series, DataFrame

img_width = 50
img_height = 50
num_classes = 2
DATA_DIR = './data/'
LABEL_CAT = 0
LABEL_DOG = 1
image_filenames = [DATA_DIR+i for i in os.listdir(DATA_DIR)] # use this for full dataset


"""
Exercise 1
"""
# Split the data in three sets, 80% for training, 10% for validation and 10% for testing
label = lambda x: LABEL_CAT if "cat" in x else LABEL_DOG
labels = [label(name) for name in image_filenames]
#print (*zip(image_filenames, labels))
xtrain, xtest, y_train, y_test = train_test_split(image_filenames, labels, test_size=0.2, random_state=123)
xval, xtest, y_val, y_test = train_test_split(xtest, y_test, test_size=0.5, random_state=1234)
# make sure that the image filenames have a fixed order before shuffling
x_train = [cv2.resize(cv2.imread(name), (img_width, img_height)) for name in xtrain]
x_val = [cv2.resize(cv2.imread(name), (img_width, img_height)) for name in xval]
x_test = [cv2.resize(cv2.imread(name), (img_width, img_height)) for name in xtest]

"""
Exercise 2
"""

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

hist = model.fit(x=np.array(x_train), y=y_train, epochs=10, verbose=1, validation_data=(np.array(x_val),y_val))

model.summary()

"""
Exercise 3
"""

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve



"""
Exercise 4
"""


"""
Exercise 5
"""


"""
Exercise 6
"""


