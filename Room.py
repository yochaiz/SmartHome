from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
import os
import matplotlib.pyplot as plt
from datetime import timedelta


class Room:
    # deviceMap = {'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe}
    deviceMap = {'LightPoints': LightPoint}

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

        for key in self.devices.keys():
            fig, ax = plt.subplots()
            ax.set_xticks([])  # clear xAxis initial values
            yLabels = []
            for i, obj in enumerate(self.devices[key]):
                # obj = self.devices[key][0]
                # stDate, seqLen = obj.findSequence(10, timedelta(minutes=2))
                # print('File:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, stDate, seqLen))
                obj.addToPlot(ax, startDate, lambdaFunc, i - .1)
                yLabels.append(obj.id)

            ax.set_yticks(range(len(self.devices[key])))
            ax.set_yticklabels(yLabels)

        plt.gcf().autofmt_xdate()
        plt.show()
