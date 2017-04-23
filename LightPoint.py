import xml.etree.ElementTree as Et
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class LightPoint(object):
    dateFormat = '%Y-%m-%d %H:%M:%S.%f'
    nullColor = 'red'
    colors = [nullColor, 'black', 'yellow']

    # nullColor = 'ro'
    # colors = [nullColor, 'ko', 'yo']

    def __init__(self, filename):
        self.filename = filename
        self.root = Et.parse(filename).getroot()
        self.id = self.root.get('Id')

    def __skipToDate(self, dstDate):
        i = 0
        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time'), LightPoint.dateFormat)
            flag = date < dstDate
            if flag is False:
                break

            i += 1

        return i

    def __plot(self, startDate, lambdaFunc, timeLocator):
        x, k = self.collectData(startDate, lambdaFunc)

        print('Start date:[%s]' % x[0].__str__())
        print('End date:[%s]' % x[len(x) - 1].__str__())
        print('nPts:[%d]' % len(x))

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(LightPoint.dateFormat))
        plt.gca().xaxis.set_major_locator(timeLocator())

        for i in range(len(x)):
            plt.plot(x[i], 0, k[i])

        plt.gcf().autofmt_xdate()
        plt.show()

    def collectData(self, startDate, lambdaFunc):
        i = self.__skipToDate(startDate)

        x = []
        k = []

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time'), LightPoint.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            x.append(date)
            i += 1

            try:
                val = int(child.text)
            except ValueError:
                val = -1

            col = LightPoint.colors[val + 1]
            k.append(col)

        return x, k

    def plotDateRange(self, startDate, endDate, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: date < endDate
        self.__plot(startDate, lambdaFunc, timeLocator)

    def plotPtsRange(self, startDate, nPts, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: len(x) < nPts
        self.__plot(startDate, lambdaFunc, timeLocator)
