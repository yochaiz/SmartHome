import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.models import Sequential
from keras.layers import LSTM, Conv2D, Reshape
import h5py
import numpy as np
from ExperimentLogger import ExperimentLogger


def loadHdf5(fname):
    f = h5py.File(fname, 'r')
    x = f['default']
    logger.info('[{}] data shape:{}'.format(fname, x.shape))
    return x


def splitData(x, y, ratio):
    i = int(x.shape[0] * ratio)
    i = x.shape[0] - i
    perm = np.random.permutation(x.shape[0])

    x = x[perm]
    xTrain = x[:i]
    xTest = x[i:]

    y = y[perm, :]
    yTrain = y[:i]
    yTest = y[i:]

    return xTrain, yTrain, xTest, yTest


x = loadHdf5('x-1-minute.h5')
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
nSamples = x.shape[0]
seqLen = x.shape[1]
nInputFeatures = x.shape[2]

y = loadHdf5('y-1-minute.h5')
y = np.array(y)
# y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
outputSize = y.shape[1]

assert (x.shape[0] == y.shape[0])  # same number of samples & labels

logger = ExperimentLogger('results/' + __file__[:-3] + '/').getLogger()

h = 4
w = 11

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(seqLen, nInputFeatures, 1), data_format='channels_last'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', data_format='channels_last'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', data_format='channels_last'))
model.add(Conv2D(1, kernel_size=(1, 1), activation='relu'))
model.add(Reshape((h, w), input_shape=(h, w, 1)))
model.add(LSTM(outputSize, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(h, w)))

model.summary(print_fn=lambda x: logger.info(x + '\n'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save(__file__[:-3] + '.h5')

xTrain, yTrain, xTest, yTest = splitData(x, y, 0.2)

nEpochs = 50
for i in range(nEpochs):
    scores = model.fit(xTrain, yTrain, epochs=1, batch_size=32, shuffle=True, validation_data=(xTest, yTest))
    logger.info('[{}]: {}'.format(i, scores.history))
    model.save(__file__[:-3] + '.h5')
