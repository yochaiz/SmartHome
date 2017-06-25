from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
from LightPoint import LightPoint
from EnergyManagement import EnergyManagement
from Plot import Plot
import os
import matplotlib.pyplot as plt


class Room:
    # deviceMap = {'LightPoints': LightPoint, 'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe,'Energy': EnergyManagement}
    # deviceMap = {'LightPoints': LightPoint, 'ThermalProbe': ThermalProbe}
    # deviceMap = {'LightPoints': LightPoint, 'AuxChannel': AuxChannel}
    deviceMap = {'AuxChannel': AuxChannel}
    # deviceMap = {'LightPoints': LightPoint}
    # deviceMap = {'ThermalProbe': ThermalProbe}
    # deviceMap = {'Energy': EnergyManagement}
    # deviceMap = {'LightPoints': LightPoint, 'Energy': EnergyManagement}

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

    def __keyDateRangeSubPlot(self, key, ax, startDate, lambdaFunc, axisMinGap):
        ax.set_xticks([])  # clear xAxis initial values
        # ax.set_yticks([])  # clear yAxis initial values
        minDate, maxDate = None, None
        for i, obj in enumerate(self.devices[key]):
            # obj = self.devices[key][0]
            # stDate, seqLen = obj.findSequence(100, timedelta(minutes=2), timedelta(hours=2))
            # print('File:[%s] - Key:[%s] - Date:[%s] - SeqLen:[%d]' % (obj.filename, key, stDate, seqLen))
            xAxisLabels, xAxisTicks = obj.addToPlot(ax, startDate, lambdaFunc, axisMinGap)
            date1 = xAxisLabels[0]
            date2 = xAxisLabels[len(xAxisLabels) - 1]
            minDate = date1 if minDate is None else min(minDate, date1)
            maxDate = date2 if maxDate is None else max(maxDate, date2)

        return xAxisLabels, xAxisTicks, minDate, maxDate

    def __genericPlot(self, dates, nPlots, iterateFunc, axisMinGap):
        # lambdaFunc = lambda x, date: date < dates[1]
        endDate = dates[1]
        lambdaFunc = lambda x, date: date if date < endDate else (
            endDate if (len(x) == 0) or (len(x) > 0 and x[len(x) - 1] < endDate) else None)

        fig, axArr = plt.subplots(nPlots, sharex=True)

        xAxisLabelsArr, xAxisTicksArr = [], []
        overallMinDate = [dates[1]]
        overallMaxDate = [dates[0]]

        def innerIterateFunc(self, j, key):
            ax = axArr[j] if nPlots > 1 else axArr
            try:
                xAxisLabels, xAxisTicks, minDate, maxDate = self.__keyDateRangeSubPlot(key, ax, dates[0], lambdaFunc,
                                                                                       axisMinGap)
                xAxisLabelsArr.append(xAxisLabels)
                xAxisTicksArr.append(xAxisTicks)
                overallMinDate[0] = min(overallMinDate[0], minDate)
                overallMaxDate[0] = max(overallMaxDate[0], maxDate)
                return ax, Plot.timedeltaToText(maxDate - minDate)

            except ValueError as e:
                return ax, e.message

        iterateFunc(self, innerIterateFunc)

        if axisMinGap is None:
            axisMinGap = round((overallMaxDate[0] - overallMinDate[0]).seconds / 80.0)

        self.__mergeSubPlotsAxis(nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, axisMinGap)

        fig.suptitle("Room:[%s]" % self.roomName, size=16)
        fig.autofmt_xdate()
        fig.subplots_adjust(hspace=0.3)
        # plt.show()

    def plotDateRange(self, startDate, endDate, axisMinGap=None):
        def iterateFunc(self, innerIterateFunc):
            for j, key in enumerate(self.devices.keys()):
                ax, timeDelta = innerIterateFunc(self, j, key)
                ax.set_title("Device:[%s] \n Time Range: %s" % (key, timeDelta))

        nPlots = len(self.devices.keys())
        self.__genericPlot([startDate, endDate], nPlots, iterateFunc, axisMinGap)

    def plotRepeatDateRange(self, startDate, endDate, key, nRepeat, timeGap, axisMinGap=None):
        dates = [startDate, endDate]
        dateFormat = '%Y-%m-%d'

        def iterateFunc(self, innerIterateFunc):
            for j in range(nRepeat):
                ax, timeDelta = innerIterateFunc(self, j, key)
                ax.set_title("Date:[%s]-[%s] \n Time Range: %s" % (
                    dates[0].strftime(dateFormat), dates[1].strftime(dateFormat), timeDelta))

                dates[0] += timeGap
                dates[1] += timeGap

        self.__genericPlot(dates, nRepeat, iterateFunc, axisMinGap)

    def __mergeSubPlotsAxis(self, nPlots, axArr, xAxisLabelsArr, xAxisTicksArr, axisMinGap):
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

                # Removes too close labels
                res = Plot.dateWithMinimalGap([xLabelsCur, xTicksCur], lambda i: xTicksCur[i] - xTicksCur[i - 1],
                                              axisMinGap)
                xLabelsCur = res[0]
                xTicksCur = res[1]

        for ax in axArr:
            ax.set_xticks(xTicksCur)
            ax.set_xticklabels(xLabelsCur)
