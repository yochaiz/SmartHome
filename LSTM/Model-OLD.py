from ExperimentLogger import ExperimentLogger
import h5py
import numpy as np


class Model:
    def __init__(self, folderName, xFname, yFname):
        self.folderName = folderName
        self.logObj = ExperimentLogger(folderName)
        self.logger = self.logObj.getLogger()

        self.x = self.loadHdf5(xFname)
        self.x = np.array(self.x)
        self.nSamples = self.x.shape[0]
        self.seqLen = self.x.shape[1]
        self.nInputFeatures = self.x.shape[2]

        self.y = self.loadHdf5(yFname)
        self.y = np.array(self.y)
        self.outputSize = self.y.shape[1]

        assert (self.x.shape[0] == self.y.shape[0])  # same number of samples & labels

        self.testRatio = -1

    def saveModel(self):
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
        self.logger.info('Training with testRatio:[{}] - nEpochs:[{}] - batchSize:[{}]'.format(testRatio, nEpochs, batchSize))
        if testRatio != self.testRatio:
            self.testRatio = testRatio
            self.xTrain, self.yTrain, self.xTest, self.yTest = self.splitData(self.x, self.y, self.testRatio)

        for i in range(nEpochs):
            scores = self.model.fit(self.xTrain, self.yTrain, epochs=1, batch_size=batchSize, shuffle=True, validation_data=(self.xTest, self.yTest))
            self.logger.info('[{}]: {}'.format(i, scores.history))
            self.saveModel()

    def loadHdf5(self, fname):
        self.logger.info('Reading [{}]'.format(fname))
        f = h5py.File(fname, 'r')
        x = f['default']
        self.logger.info('[{}] data shape:{}'.format(fname, x.shape))
        return x

    def splitData(self, x, y, ratio):
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
