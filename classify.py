#coding:utf-8
# @makai
# 2016/05/01

from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np


#load data
def load_data(training):

    data = None
    label = None
    filedir = ''

    if training:
        filedir = './data_training_pengzhang'
    else:
        filedir = './data_test_txt'

    filelist = os.listdir(filedir)
    imagelist = []
    for filename in filelist:
        splits = str(filename).split('.')
        if splits[-1] == 'png':
            imagelist.append(filename)

    imageCount = len(imagelist)
    data = np.empty((imageCount, 1, 65, 65), dtype="float32")
    label = np.empty((imageCount,), dtype="uint8")
    j = 0
    for filename in imagelist:
        splits = str(filename).split('.')
        img = Image.open(filedir + "/" + filename)
        arr = np.asarray(img, dtype='float32')
        data[j, :, :, :] = arr
        label[j] = int(filename.split('_')[0])
        j += 1
    return data, label


def getModel():
    #construct cnn model
    model = Sequential()

    #first layer, 6 con cores
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(1, 65, 65)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #second layer, 8 con cores
    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #first full connection layer
    model.add(Flatten())
    model.add(Dense(output_dim=120))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim=200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #Softmax output layer, 10 classes
    model.add(Dense(output_dim=62))
    model.add(Activation('softmax'))

    #prepare to training or test
    #sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def train(model, data, label):
    #begin to train
    label = np_utils.to_categorical(label, 62)
    model.fit(data, label, batch_size=64, nb_epoch=10, shuffle=True, verbose=1, validation_split=0)
    model.save_weights('Lenet_pengzhang.hdf5', overwrite=True)


def test(model, data, label):
    model.load_weights('Lenet_pengzhang.hdf5')
    #predict
    outcome = model.predict_classes(data, batch_size=32, verbose=1)
    print("real class:")
    print(label)
    print("predict class:")
    print(outcome)
    allCount = len(label)
    rightCount = 0
    for x in xrange(0, len(label)):
        if outcome[x] == label[x]:
            rightCount += 1
    print('\nallCount = ' + str(allCount) + ', rightCount = ' + str(rightCount) + ', acc = ' + str(float(rightCount) / allCount))
    # print "allCount = " + sr(allCount) + ', rightCount = ' + str(rightCount) + ', acc = ' + str(float(rightCount) / allCount)


cmd = 'matlab -nodesktop -nosplash -r trans'
os.system(cmd)
TRAINING_NOW = False
data, label = load_data(TRAINING_NOW)
data = data.reshape(data.shape[0], 1, 65, 65)
data = data.astype('float32')
data /= 255

test(getModel(), data, label)
