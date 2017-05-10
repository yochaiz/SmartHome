from abc import ABCMeta, abstractmethod
import xml.etree.ElementTree as Et
from datetime import datetime
import matplotlib.pyplot as plt
from Plot import Plot


class Device:
    __metaclass__ = ABCMeta

    dateFormat = '%Y-%m-%d %H:%M:%S'

    def __init__(self, filename):
        self.filename = filename
        self.root = Et.parse(filename).getroot()
        self.id = self.root.get('Id')

    def __skipToDate(self, dstDate):
        i = 0
        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], Device.dateFormat)
            flag = date < dstDate
            if flag is False:
                break

            i += 1

        return i

    @abstractmethod
    def collectData(self, startDate, lambdaFunc):
        raise NotImplementedError('subclasses must override collectData()!')

    @abstractmethod
    def __plotInternal(self, ax, x, k, pos=None):
        raise NotImplementedError('subclasses must override __plot()!')

    @abstractmethod
    def buildPlotData(self, x, k):
        raise NotImplementedError('subclasses must override buildPlotData()!')

    def __sortPlotXaxis(self, ax, x, xAxisLabels=None, xAxisTicks=None):
        [xAxis] = Plot.dateWithMinimalGap([x], lambda i: (x[i] - x[i - 1]).seconds)

        ax.set_xticks(xAxis)
        ax.set_xticklabels(xAxis)

    def addToPlot(self, ax, startDate, lambdaFunc, pos=0):
        x, k = self.collectData(startDate, lambdaFunc)

        print('Start date:[%s]' % x[0])
        print('End date:[%s]' % x[len(x) - 1])
        print('nPts:[%d]' % len(x))

        xAxisLabels, xAxisTicks = self.__plotInternal(ax, x, k, pos)

        self.__sortPlotXaxis(ax, x, xAxisLabels, xAxisTicks)
        # return x[0], x[len(x) - 1]
        return xAxisLabels, xAxisTicks

    def __plot(self, startDate, lambdaFunc):
        fig, ax = plt.subplots()
        ax.set_xticks([])  # clear xAxis initial values
        xAxisLabels, xAxisTicks = self.addToPlot(ax, startDate, lambdaFunc)
        date1 = xAxisLabels[0]
        date2 = xAxisLabels[len(xAxisLabels) - 1]
        ax.set_title("[%s] \n Time Range: %s" % (self.filename, Plot.timedeltaToText(date2 - date1)))
        plt.gcf().autofmt_xdate()
        plt.show()

    def plotDateRange(self, startDate, endDate):
        lambdaFunc = lambda x, date: date < endDate
        self.__plot(startDate, lambdaFunc)

    def plotPtsRange(self, startDate, nPts):
        lambdaFunc = lambda x, date: len(x) < nPts
        self.__plot(startDate, lambdaFunc)

    # finds sequence in seqLen length with minGap between two neighbor elements
    # if there is no seqLen length sequence, returns the longest sequence exists
    def findSequence(self, seqLen, minGap):
        i = 0
        curSeqLen = 0
        startDate = None
        curDate = None
        maxSeqLen = 0
        maxSeqStartDate = None
        while i < len(self.root) and curSeqLen < seqLen:
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], Device.dateFormat)

            if curSeqLen == 0:
                startDate = date
                curDate = date
                curSeqLen += 1
            elif date - curDate >= minGap:
                curSeqLen += 1
                curDate = date
            else:
                curSeqLen = 1
                startDate = date
                curDate = date

            if curSeqLen > maxSeqLen:
                maxSeqLen = curSeqLen
                maxSeqStartDate = startDate

            i += 1

        if curSeqLen == seqLen:
            return startDate, seqLen

        return maxSeqStartDate, maxSeqLen
