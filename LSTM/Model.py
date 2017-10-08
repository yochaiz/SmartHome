from ExperimentLogger import ExperimentLogger
import h5py
import numpy as np
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import json


class Model:
    def __init__(self, folderName, dataFolderName, dataPrepFunc, gpuNum, gpuFraction):
        self.folderName = folderName
        self.dataFolderName = dataFolderName
        self.logObj = ExperimentLogger(folderName)
        self.logger = self.logObj.getLogger()

        self.dataPrepFunc = dataPrepFunc
        self.fileIdx = 1
        self.x, self.y = self.loadData()

        self.seqLen = self.x.shape[1]
        self.nInputFeatures = self.x.shape[2]
        self.outputSize = self.y.shape[1]

        assert (self.x.shape[0] == self.y.shape[0])  # same number of samples & labels

        self.gpuNum = gpuNum
        self.gpuFraction = gpuFraction

        self.model = None
        self.perm = None

        self.createConfigFile()

    def createConfigFile(self):
        config = {}
        config['dataFolderName'] = self.dataFolderName
        with open('config.json', 'w') as f:
            json.dump(config, f)

    def loadData(self):
        xFname = '{}/x-{}.h5'.format(self.dataFolderName, self.fileIdx)
        yFname = '{}/y-{}.h5'.format(self.dataFolderName, self.fileIdx)

        x = None
        y = None

        if os.path.exists(xFname) and os.path.exists(yFname):
            # self.logger.info('Loading [{}] & [{}]'.format(xFname, yFname))
            x = self.loadHdf5(xFname)
            x = np.array(x)

            y = self.loadHdf5(yFname)
            y = np.array(y)

            if self.dataPrepFunc is not None:
                x, y = self.dataPrepFunc(x, y)

            self.fileIdx += 1

        return x, y

    def saveModel(self):
        if self.model is not None:
            self.model.save(self.logObj.rootDir + self.logObj.dirName + '/' + self.modelFname + '.h5')

    def modifyModelFname(self):
        idx = self.modelFname.rfind('/')
        if idx >= 0:
            self.modelFname = self.modelFname[idx + 1:]

    def setModel(self, model, modelFname):
        self.modelFname = modelFname
        self.modifyModelFname()

        self.model = model
        self.model.summary(print_fn=lambda x: self.logger.info(x))
        self.model.summary()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.saveModel()

    def train(self, testRatio, nEpochs, batchSize):
        # limit memory precentage usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpuFraction
        set_session(tf.Session(config=config))

        self.logger.info('Training with testRatio:[{}] - nEpochs:[{}] - batchSize:[{}]'.format(testRatio, nEpochs, batchSize))

        opt = {'loss': None, 'Epoch': -1}
        for i in range(nEpochs):
            avgLoss = {'train': 0.0, 'test': 0.0}

            while (self.x is not None) and (self.y is not None):
                self.xTrain, self.yTrain, self.xTest, self.yTest = self.splitData(self.x, self.y, testRatio)
                scores = self.model.fit(self.xTrain, self.yTrain, epochs=1, batch_size=batchSize, shuffle=True, validation_data=(self.xTest, self.yTest))

                avgLoss['train'] += scores.history['loss'][0]
                avgLoss['test'] += scores.history['val_loss'][0]

                self.logger.info('[{}]: {}'.format(i, scores.history))
                self.saveModel()

                self.x, self.y = self.loadData()

            avgLoss['train'] /= (self.fileIdx - 1)
            avgLoss['test'] /= (self.fileIdx - 1)
            self.logger.info('[{}]: Average loss {}'.format(i, avgLoss))

            if (opt['loss'] is None) or (avgLoss['train'] < opt['loss']):
                self.logger.info('opt: {}'.format(opt))
                opt['loss'] = avgLoss['train']
                opt['Epoch'] = i
                self.model.save(self.logObj.rootDir + self.logObj.dirName + '/opt-' + self.modelFname + '.h5')

            self.fileIdx = 1
            self.x, self.y = self.loadData()

    def loadHdf5(self, fname):
        self.logger.info('Reading [{}]'.format(fname))
        f = h5py.File(fname, 'r')
        x = f['default']
        self.logger.info('[{}] data shape:{}'.format(fname, x.shape))
        return x

    def splitData(self, x, y, ratio):
        i = int(x.shape[0] * ratio)
        i = x.shape[0] - i

        # use same permutation for all dataset chunks
        # in order to assure sample is either for training or for test
        # but NOT one epoch for training and on another epoch for test
        if self.perm is None:
            self.perm = np.random.permutation(x.shape[0])

        x = x[self.perm]
        xTrain = x[:i]
        xTest = x[i:]

        y = y[self.perm, :]
        yTrain = y[:i]
        yTest = y[i:]

        return xTrain, yTrain, xTest, yTest
