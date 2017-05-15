from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
from Plot import Plot
import os
import matplotlib.pyplot as plt
from datetime import timedelta


class Room:
    # deviceMap = {'LightPoints': LightPoint, 'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe}
    deviceMap = {'LightPoints': LightPoint, 'ThermalProbe': ThermalProbe}

    # deviceMap = {'LightPoints': LightPoint, 'AuxChannel': AuxChannel}
    # deviceMap = {'LightPoints': LightPoint}
    # deviceMap = {'ThermalProbe': ThermalProbe}

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

    def __keyDateRangeSubPlot(self, key, ax, startDate, lambdaFunc):
        ax.set_xticks([])  # clear xAxis initial values
        ax.set_yticks([])  # clear yAxis initial values
        minDate, maxDate = None, None
        for i, obj in enumerate(self.devices[key]):
            # obj = self.devices[key][0]
            # stDate, seqLen = obj.findSequence(100, timedelta(minutes=2))
            # print('File:[%s] - Key:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, key, stDate, seqLen))
            xAxisLabels, xAxisTicks = obj.addToPlot(ax, startDate, lambdaFunc)
            date1 = xAxisLabels[0]
            date2 = xAxisLabels[len(xAxisLabels) - 1]
            minDate = date1 if minDate is None else min(minDate, date1)
            maxDate = date2 if maxDate is None else max(maxDate, date2)

        return xAxisLabels, xAxisTicks, minDate, maxDate

    def __genericPlot(self, dates, nPlots, iterateFunc):
        # lambdaFunc = lambda x, date: date < dates[1]
        endDate = dates[1]
        lambdaFunc = lambda x, date: date if date < endDate else (
            endDate if (len(x) == 0) or (len(x) > 0 and x[len(x) - 1] < endDate) else None)

        fig, axArr = plt.subplots(nPlots)
        xAxisLabelsArr, xAxisTicksArr = [], []
        overallMinDate = [dates[1]]
        overallMaxDate = [dates[0]]

        def innerIterateFunc(self, j, key):
            ax = axArr[j] if nPlots > 1 else axArr
            xAxisLabels, xAxisTicks, minDate, maxDate = self.__keyDateRangeSubPlot(key, ax, dates[0], lambdaFunc)
            xAxisLabelsArr.append(xAxisLabels)
            xAxisTicksArr.append(xAxisTicks)
            overallMinDate[0] = min(overallMinDate[0], minDate)
            overallMaxDate[0] = max(overallMaxDate[0], maxDate)
            return ax, maxDate - minDate

        iterateFunc(self, innerIterateFunc)

        minGap = round((overallMaxDate[0] - overallMinDate[0]).seconds / 80.0)
        self.__mergeSubPlotsAxis(nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, minGap)

        fig.suptitle("Room:[%s]" % self.roomName, size=16)
        fig.autofmt_xdate()
        plt.subplots_adjust(hspace=0.3)
        plt.show()

    def plotDateRange(self, startDate, endDate):
        def iterateFunc(self, innerIterateFunc):
            for j, key in enumerate(self.devices.keys()):
                ax, timeDelta = innerIterateFunc(self, j, key)
                ax.set_title("Device:[%s] \n Time Range: %s" % (key, Plot.timedeltaToText(timeDelta)))

        nPlots = len(self.devices.keys())
        self.__genericPlot([startDate, endDate], nPlots, iterateFunc)

    def plotRepeatDateRange(self, startDate, endDate, key, nRepeat, timeGap):
        dates = [startDate, endDate]
        dateFormat = '%Y-%m-%d'

        def iterateFunc(self, innerIterateFunc):
            for j in range(nRepeat):
                ax, timeDelta = innerIterateFunc(self, j, key)
                ax.set_title("Date:[%s]-[%s] \n Time Range: %s" % (
                    dates[0].strftime(dateFormat), dates[1].strftime(dateFormat), Plot.timedeltaToText(timeDelta)))
                dates[0] += timeGap
                dates[1] += timeGap

        self.__genericPlot(dates, nRepeat, iterateFunc)

    def __mergeSubPlotsAxis(self, nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, minGap):
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

                i += 1

            res = Plot.dateWithMinimalGap([xLabelsCur, xTicksCur], lambda i: xTicksCur[i] - xTicksCur[i - 1], minGap)
            xLabelsCur = res[0]
            xTicksCur = res[1]

        for ax in axArr:
            ax.set_xticks(xTicksCur)
            ax.set_xticklabels(xLabelsCur)
