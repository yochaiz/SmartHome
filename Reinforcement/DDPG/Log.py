# base class that applies some log behaviour for sub-classes
from abc import ABCMeta, abstractmethod


class Log:
    __metaclass__ = ABCMeta

    dictTypes = [str, int, float]
    modelAttrName = 'model'

    # list of all Log objs
    objs = []

    def __init__(self, nBackups):
        self.nBackups = nBackups
        self.curBackupIdx = 0

        # add self to objects list
        Log.objs.append(self)

    def className(self):
        return self.__class__.__name__

    def __save(self, dirName, logger):
        # build full path
        fullPath = '{}/model-{}-{}.h5'.format(dirName, self.className(), self.curBackupIdx)
        # log model file name
        logger.info('Saving [{}] model as [{}]'.format(self.className(), fullPath))
        # save model
        self.model.save(fullPath)
        # update next save index
        self.curBackupIdx = (self.curBackupIdx + 1) % self.nBackups

    # save function for all list objects
    @staticmethod
    def save(dirName, logger):
        for obj in Log.objs:
            if hasattr(obj, Log.modelAttrName):
                obj.__save(dirName, logger)

    def __printModel(self, logger):
        if hasattr(self, self.modelAttrName):
            logger.info('[{}] model architecture:'.format(self.className()))
            logger.info('============================')
            self.model.summary(print_fn=lambda x: logger.info(x))

    # print model function for all list objects
    @staticmethod
    def printModel(logger):
        for obj in Log.objs:
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
        for obj in Log.objs:
            dict[obj.className()] = obj.__toJSON()

        return dict
