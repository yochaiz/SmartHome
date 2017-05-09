from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
from Plot import Plot
import os
import matplotlib.pyplot as plt
from datetime import timedelta


class Room:
    # deviceMap = {'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe}
    deviceMap = {'LightPoints': LightPoint, 'AuxChannel': AuxChannel}

    # deviceMap = {'AuxChannel': AuxChannel}

    def __init__(self, roomFolderName):
        self.roomName = roomFolderName
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

        nPlots = len(self.devices.keys())
        fig, axArr = plt.subplots(nPlots)
        for j, key in enumerate(self.devices.keys()):
            ax = axArr[j] if nPlots > 1 else axArr
            ax.set_xticks([])  # clear xAxis initial values
            yLabels = []
            minDate = endDate
            maxDate = startDate
            for i, obj in enumerate(self.devices[key]):
                # obj = self.devices[key][0]
                # stDate, seqLen = obj.findSequence(10, timedelta(minutes=2))
                # print('File:[%s] - Key:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, key, stDate, seqLen))
                date1, date2 = obj.addToPlot(ax, startDate, lambdaFunc, i - .1)
                minDate = min(minDate, date1)
                maxDate = max(maxDate, date2)
                yLabels.append(obj.id)

            ax.set_yticks(range(len(self.devices[key])))
            ax.set_yticklabels(yLabels)
            ax.set_title("Device:[%s] \n Time Range: %s" % (key, Plot.timedeltaToText(maxDate - minDate)))

        fig.suptitle("Room:[%s]" % self.roomName, size=16)
        # fig.subplots_adjust(hspace=1.5)
        fig.autofmt_xdate()
        plt.show()
