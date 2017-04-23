from os import listdir
from LightPoint import LightPoint


class RoomLights(object):
    dataFolder = 'data/'
    LightPointsFolder = dataFolder + 'LightPoints/'
    filePrefix = '.LightPoint.'
    fileSuffix = '.'

    def __init__(self, roomNumber):
        self.number = roomNumber
        self.LightPoints = []

        for filename in listdir(RoomLights.LightPointsFolder):
            startPos = filename.find(RoomLights.filePrefix)
            if startPos >= 0:
                startPos += len(RoomLights.filePrefix)
                endPos = filename.find(RoomLights.fileSuffix, startPos)
                if endPos >= 0:
                    rNum = filename[startPos:endPos]
                    if rNum == roomNumber:
                        self.LightPoints.append(LightPoint(RoomLights.LightPointsFolder + filename))

        print('Room [%s] contains [%d] lights' % (roomNumber, len(self.LightPoints)))
