from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
import os
import matplotlib.pyplot as plt
from datetime import timedelta


class Room:
    # deviceMap = {'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe}
    deviceMap = {'AuxChannel': AuxChannel}

    def __init__(self, roomFolderName):
        self.devices = {}
        for key in self.deviceMap.keys():
            folderPath = roomFolderName + '/' + key
            if os.path.isdir(folderPath):
                self.devices[key] = []
                for filename in os.listdir(folderPath):
                    if filename.endswith(".xml"):
                        # creates object for each file
                        self.devices[key].append(self.deviceMap[key](folderPath + '/' + filename))

    def plotDateRange(self, startDate, endDate):
        lambdaFunc = lambda x, date: date < endDate
        fig, ax = plt.subplots()

        for key in self.devices.keys():
            for obj in self.devices[key]:
                stDate, seqLen = obj.findSequence(10, timedelta(minutes=2))
                print('File:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, stDate, seqLen))
                # obj.addToPlot(ax, startDate, lambdaFunc)

        plt.gcf().autofmt_xdate()
        plt.show()
