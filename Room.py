from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
from Plot import Plot
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np


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

        minGap = round((maxDate - minDate).seconds / 80.0)
        self.mergeSubPlotsAxis(axArr, minGap)

        fig.suptitle("Room:[%s]" % self.roomName, size=16)
        # fig.subplots_adjust(hspace=1.5)
        fig.autofmt_xdate()
        plt.show()

    def mergeSubPlotsAxis(self, axArr, minGap):
        if len(axArr) <= 1:
            return

        xticks = []
        xticklabels = []
        for ax in axArr:
            xticks.append(ax.get_xticks().tolist())
            xticklabels.append(ax.get_xticklabels())

        xTicksCur = xticks[0]
        xLabelsCur = xticklabels[0]

        for z in range(1, len(xticks)):
            i = 0
            j = 0
            axTicks = xticks[z]
            axLabels = xticklabels[z]
            while i < len(axTicks):
                while j < len(xTicksCur) and axTicks[i] > xTicksCur[j]:
                    j += 1

                if j >= len(xTicksCur) or axTicks[i] < xTicksCur[j]:
                    xTicksCur.insert(j, axTicks[i])
                    xLabelsCur.insert(j, axLabels[i])
                    # xLabelsCur = np.insert(xLabelsCur, j, axLabels[i])

                i += 1

            res = Plot.dateWithMinimalGap([xLabelsCur, xTicksCur], lambda i: xTicksCur[i] - xTicksCur[i - 1], minGap)
            xLabelsCur = res[0]
            xTicksCur = res[1]

        for ax in axArr:
            ax.set_xticks(xTicksCur)
            ax.set_xticklabels(xLabelsCur)
