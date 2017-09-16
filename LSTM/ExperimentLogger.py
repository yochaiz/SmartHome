import logging
import os
from datetime import datetime


class ExperimentLogger(object):
    # rootDir = 'results/'

    def __init__(self, rootDir):
        self.rootDir = rootDir


    def getLogger(self):
        now = datetime.now()
        self.dirName = 'D-{}-{}-H-{}-{}'.format(now.day, now.month, now.hour, now.minute)
        if not os.path.exists(self.rootDir + self.dirName):
            os.makedirs(self.rootDir + self.dirName)

        # initialize logger
        logging.basicConfig(level=logging.INFO, filename=self.rootDir + self.dirName + '/info.log')
        return logging.getLogger(__name__)
