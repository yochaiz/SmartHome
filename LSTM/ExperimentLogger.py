import logging
import os
from datetime import datetime


class ExperimentLogger(object):
    # rootDir = 'results/'

    def __init__(self, rootDir):
        self.rootDir = rootDir

    def __getLogger(self, folderName):
        if not os.path.exists(folderName):
            os.makedirs(folderName)

        # initialize logger
        logging.basicConfig(level=logging.INFO, filename=folderName + '/info.log')
        return logging.getLogger(__name__)

    def getBasicLogger(self):
        return self.__getLogger(self.rootDir)

    def getLoggerWithTime(self):
        now = datetime.now()
        dirName = 'D-{}-{}-H-{}-{}'.format(now.day, now.month, now.hour, now.minute)

        return self.__getLogger(self.rootDir + dirName)
