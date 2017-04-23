import xml.etree.ElementTree as Et
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class LightPoint(object):
    dateFormat = '%Y-%m-%d %H:%M:%S.%f'
    nullColor = 'ro'
    colors = [nullColor, 'ko', 'yo']

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

    @staticmethod
    def plot(x, k, timeLocator):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(LightPoint.dateFormat))
        plt.gca().xaxis.set_major_locator(timeLocator())

        for i in range(len(x)):
            plt.plot(x[i], 0, k[i])

        plt.gcf().autofmt_xdate()
        plt.show()

    def plotDateRange(self, startDate, endDate, timeLocator=mdates.HourLocator):
        i = self.__skipToDate(startDate)

        x = []
        k = []

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time'), LightPoint.dateFormat)
            flag = date < endDate
            if flag is False:
                break

            x.append(date)
            print(date)
            i += 1

            try:
                val = int(child.text)
            except ValueError:
                val = -1

            col = LightPoint.colors[val + 1]
            k.append(col)

        print('Start date:[%s]' % x[0].__str__())
        print('End date:[%s]' % x[len(x) - 1].__str__())
        print('nPts:[%d]' % len(x))

        LightPoint.plot(x, k, timeLocator)

    def plotPtsRange(self, startDate, nPts, timeLocator=mdates.HourLocator):
        i = self.__skipToDate(startDate)

        x = []
        k = []

        maxIdx = min(i + nPts, len(self.root))
        while i < maxIdx:
            child = self.root[i]
            date = datetime.strptime(child.get('Time'), LightPoint.dateFormat)
            x.append(date)
            print(date)
            i += 1

            try:
                val = int(child.text)
            except ValueError:
                val = -1

            col = LightPoint.colors[val + 1]
            k.append(col)

        print('Start date:[%s]' % x[0].__str__())
        print('End date:[%s]' % x[len(x) - 1].__str__())
        print('nPts:[%d]' % len(x))

        LightPoint.plot(x, k, timeLocator)
