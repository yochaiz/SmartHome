import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.models import Sequential
from keras.layers import LSTM, Conv2D, Reshape
from ..Model import Model

folderName = 'results/' + __file__[:-3] + '/'
m = Model(folderName, 'x-1-minute.h5', 'y-1-minute.h5')


m.x = m.x.reshape(m.x.shape[0], m.x.shape[1], m.x.shape[2], 1)
# y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

h = 4
w = 11

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(m.seqLen, m.nInputFeatures, 1), data_format='channels_last'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', data_format='channels_last'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', data_format='channels_last'))
model.add(Conv2D(1, kernel_size=(1, 1), activation='relu'))
model.add(Reshape((h, w), input_shape=(h, w, 1)))
model.add(LSTM(m.outputSize, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(h, w)))

m.setModel(model, __file__[:-3])

nEpochs = 50
testRatio = 0.2
batchSize = 32
m.train(testRatio, nEpochs, batchSize)
