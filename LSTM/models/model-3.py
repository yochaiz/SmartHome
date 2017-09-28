import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.models import Sequential
from keras.layers import LSTM
from Model import Model

folderName = 'results/' + __file__[:-3] + '/'
m = Model(folderName, 'x-1-second.h5', 'y-1-second.h5')

model = Sequential()
model.add(LSTM(m.outputSize, activation='sigmoid', dropout=0.3, recurrent_dropout=0.3, input_shape=(m.seqLen, m.nInputFeatures)))

m.setModel(model, __file__[:-3])

nEpochs = 50
testRatio = 0.2
batchSize = 128
m.train(testRatio, nEpochs, batchSize)