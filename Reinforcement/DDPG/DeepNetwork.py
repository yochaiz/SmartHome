# base class that applies some log behaviour for sub-classes
from abc import ABCMeta, abstractmethod


class DeepNetwork:
    __metaclass__ = ABCMeta

    dictTypes = [str, int, float]
    trainModelKey = 'train'
    targetModelKey = 'target'

    # list of all Log objs
    objs = []

    def __init__(self, seqLen, stateDim, actionDim, TAU, nBackups):
        self.nBackups = nBackups
        self.curBackupIdx = 0

        self.seqLen = seqLen
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.TAU = TAU

        # create models
        self.models = {self.trainModelKey: None, self.targetModelKey: None}
        # # create training model
        # self.models[self.trainModelAttrName] = self.buildModel()
        # # create target (final) model
        # self.models[self.targetModelKey] = self.buildModel()

        # add self to objects list
        DeepNetwork.objs.append(self)

    # update target model SLOWLY by current trained model
    def updateModel(self):
        wModel = self.models[self.trainModelKey].get_weights()
        wTargetModel = self.models[self.targetModelKey].get_weights()
        assert (len(wModel) == len(wTargetModel))
        for i in xrange(len(wTargetModel)):
            wTargetModel[i] = (self.TAU * wModel[i]) + ((1 - self.TAU) * wTargetModel[i])

    def className(self):
        return self.__class__.__name__

    def __save(self, dirName, logger):
        for key in self.models.iterkeys():
            if self.models[key] is None:
                continue

            # build full path
            fullPath = '{}/{}-{}-model-{}.h5'.format(dirName, self.className(), key, self.curBackupIdx)
            # log model file name
            logger.info('Saving [{}] {} model as [{}]'.format(self.className(), key, fullPath))
            # save model
            self.models[key].save(fullPath)

        # update next save index
        self.curBackupIdx = (self.curBackupIdx + 1) % self.nBackups

    # save function for all list objects
    @staticmethod
    def save(dirName, logger):
        for obj in DeepNetwork.objs:
            obj.__save(dirName, logger)

    def __printModel(self, logger):
        if self.models[self.trainModelKey] is not None:
            logger.info('[{}] model architecture:'.format(self.className()))
            logger.info('============================')
            self.models[self.trainModelKey].summary(print_fn=lambda x: logger.info(x))

    # print model function for all list objects
    @staticmethod
    def printModel(logger):
        for obj in DeepNetwork.objs:
            obj.__printModel(logger)

    # convert class object to JSON serializable
    def __toJSON(self):
        var = dict(vars(self))  # make dict copy
        keysToDelete = []
        for key, val in var.iteritems():
            if type(val) not in self.dictTypes:
                keysToDelete.append(key)

        for key in keysToDelete:
            del var[key]

        return var

    # convert to JSON function for all list objects
    @staticmethod
    def toJSON():
        dict = {}
        for obj in DeepNetwork.objs:
            dict[obj.className()] = obj.__toJSON()

        return dict
