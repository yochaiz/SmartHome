from abc import ABCMeta, abstractmethod
import xml.etree.ElementTree as Et
from datetime import datetime
import matplotlib.dates as mdates


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
    def __plot(self, startDate, lambdaFunc, timeLocator):
        raise NotImplementedError('subclasses must override __plot()!')

    def plotDateRange(self, startDate, endDate, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: date < endDate
        self.__plot(startDate, lambdaFunc, timeLocator)

    def plotPtsRange(self, startDate, nPts, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: len(x) < nPts
        self.__plot(startDate, lambdaFunc, timeLocator)
