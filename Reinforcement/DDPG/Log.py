# base class that applies some log behaviour for sub-classes
from abc import ABCMeta, abstractmethod


class Log:
    __metaclass__ = ABCMeta

    dictTypes = [str, int, float]

    def __init__(self, nBackups):
        self.nBackups = nBackups
        self.curBackupIdx = 0

    def className(self):
        return self.__class__.__name__

    def save(self, dirName, logger):
        # build full path
        fullPath = '{}/model-{}-{}.h5'.format(dirName, self.className(), self.curBackupIdx)
        # log model file name
        logger.info('Saving [{}] model as [{}]'.format(self.className(), fullPath))
        # save model
        self.model.save(fullPath)
        # update next save index
        self.curBackupIdx = (self.curBackupIdx + 1) % self.nBackups

    def printModel(self, logger):
        if self.model is not None:
            logger.info('[{}] model architecture:'.format(self.className()))
            logger.info('============================')
            self.model.summary(print_fn=lambda x: logger.info(x))

    # convert class object to JSON serializable
    def toJSON(self):
        var = dict(vars(self))  # make dict copy
        keysToDelete = []
        for key, val in var.iteritems():
            if type(val) not in self.dictTypes:
                keysToDelete.append(key)

        for key in keysToDelete:
            del var[key]

        return var
