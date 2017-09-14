from keras.models import Sequential
from keras.layers import LSTM
import h5py
import os
import numpy as np
from ExperimentLogger import ExperimentLogger

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def loadHdf5(fname):
    f = h5py.File(fname, 'r')
    x = f['default']
    print('data shape:%s' % str(x.shape))
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


x = loadHdf5('x.hdf5')
x = np.array(x)
nSamples = x.shape[0]
seqLen = x.shape[1]
nInputFeatures = x.shape[2]

y = loadHdf5('y.hdf5')
y = np.array(y)
outputSize = y.shape[1]

assert (x.shape[0] == y.shape[0])  # same number of samples & labels

logger = ExperimentLogger().getLogger()

model = Sequential()
model.add(LSTM(outputSize, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(seqLen, nInputFeatures)))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save(__file__[:-3] + '.h5')

xTrain, yTrain, xTest, yTest = splitData(x, y, 0.2)

scores = model.fit(xTrain, yTrain, epochs=1, batch_size=32, shuffle=True, validation_data=(xTest, yTest))
logger.info(scores.history)
model.save(__file__[:-3] + '.h5')
