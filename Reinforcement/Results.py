from datetime import datetime
import os
import shutil


class Results:
    rootDir = 'results'
    endedFolder = 'ended'

    def __init__(self, baseFolder, actionDim):
        self.loss = []
        self.score = []

        self.baseFolder = baseFolder
        self.actionDim = 'actionDim-{}'.format(actionDim)
        now = datetime.now()
        self.folderName = 'D-{}-{}-H-{}-{}-{}'.format(now.day, now.month, now.hour, now.minute, now.second)
        self.fullPath = '{}/{}/{}/{}'.format(baseFolder, self.rootDir, self.actionDim, self.folderName)

        if not os.path.exists(self.fullPath):
            os.makedirs(self.fullPath)

    def getFullPath(self):
        return self.fullPath

    def moveToEnded(self):
        newFullPath = '{}/{}/{}/{}/{}'.format(self.baseFolder, self.rootDir, self.actionDim, self.endedFolder,
                                              self.folderName)
        shutil.move(self.fullPath, newFullPath)
        self.fullPath = newFullPath

    def toJSON(self):
        return dict(vars(self))  # make dict copy
