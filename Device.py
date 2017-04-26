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
    def __plotInternal(self, ax, x, k):
        raise NotImplementedError('subclasses must override __plot()!')

    def plot(self, startDate, lambdaFunc):
        x, k = self.collectData(startDate, lambdaFunc)

        print('Start date:[%s]' % x[0])
        print('End date:[%s]' % x[len(x) - 1])
        print('nPts:[%d]' % len(x))

        fig, ax = plt.subplots()

        self.__plotInternal(ax, x, k)
        ax.set_title("[%s] \n Time Range: %s" % (self.filename, Plot.timedeltaToText(x[len(x) - 1] - x[0])))

        [xAxis] = Plot.dateWithMinimalGap([x], lambda i: (x[i] - x[i - 1]).seconds)

        ax.set_xticks(xAxis)
        ax.set_xticklabels(xAxis)

        plt.gcf().autofmt_xdate()
        plt.show()

    def plotDateRange(self, startDate, endDate):
        lambdaFunc = lambda x, date: date < endDate
        self.plot(startDate, lambdaFunc)

    def plotPtsRange(self, startDate, nPts):
        lambdaFunc = lambda x, date: len(x) < nPts
        self.plot(startDate, lambdaFunc)
