from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
from Plot import Plot
import os
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


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

    def keyDateRangeSubPlot(self, key, ax, startDate, lambdaFunc):
        ax.set_xticks([])  # clear xAxis initial values
        yLabels = []
        minDate, maxDate = None, None
        for i, obj in enumerate(self.devices[key]):
            # obj = self.devices[key][0]
            # stDate, seqLen = obj.findSequence(10, timedelta(minutes=2))
            # print('File:[%s] - Key:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, key, stDate, seqLen))
            xAxisLabels, xAxisTicks = obj.addToPlot(ax, startDate, lambdaFunc, i - .1)
            date1 = xAxisLabels[0]
            date2 = xAxisLabels[len(xAxisLabels) - 1]
            minDate = date1 if minDate is None else min(minDate, date1)
            maxDate = date2 if maxDate is None else max(maxDate, date2)
            yLabels.append(obj.id)

        ax.set_yticks(range(len(self.devices[key])))
        ax.set_yticklabels(yLabels)
        ax.set_title("Device:[%s] \n Time Range: %s" % (key, Plot.timedeltaToText(maxDate - minDate)))

        return xAxisLabels, xAxisTicks, minDate, maxDate

    def plotDateRange(self, startDate, endDate):
        lambdaFunc = lambda x, date: date < endDate

        nPlots = len(self.devices.keys())
        fig, axArr = plt.subplots(nPlots)
        xAxisLabelsArr, xAxisTicksArr = [], []
        overallMinDate = endDate
        overallMaxDate = startDate
        for j, key in enumerate(self.devices.keys()):
            ax = axArr[j] if nPlots > 1 else axArr
            xAxisLabels, xAxisTicks, minDate, maxDate = self.keyDateRangeSubPlot(key, ax, startDate, lambdaFunc)
            xAxisLabelsArr.append(xAxisLabels)
            xAxisTicksArr.append(xAxisTicks)
            overallMinDate = min(overallMinDate, minDate)
            overallMaxDate = max(overallMaxDate, maxDate)

        minGap = round((overallMaxDate - overallMinDate).seconds / 80.0)
        self.mergeSubPlotsAxis(nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, minGap)

        for ax in axArr:
            jj = ax.get_xticklabels()

        fig.suptitle("Room:[%s]" % self.roomName, size=16)
        # fig.subplots_adjust(hspace=1.5)
        fig.autofmt_xdate()
        plt.show()

    def plotRepeatDateRange(self, startDate, endDate, key, nRepeat):
        if nRepeat <= 0:
            return

    def mergeSubPlotsAxis(self, nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, minGap):
        if nPlots <= 1:
            return

        xTicksCur = xAxisTicksArr[0]
        xLabelsCur = xAxisLabelsArr[0]

        for z in range(1, len(xAxisTicksArr)):
            i = 0
            j = 0
            axTicks = xAxisTicksArr[z]
            axLabels = xAxisLabelsArr[z]
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
