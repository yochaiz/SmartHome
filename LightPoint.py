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

    def plot(self, startDate, endDate, timeLocator=mdates.HourLocator):
        flag = True
        i = 0
        while flag and i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time'), LightPoint.dateFormat)
            flag = date < startDate
            i += 1

        i -= 1
        print('Start date:[%s]' % date.__str__())

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

        print('End date:[%s]' % date.__str__())
        print('nPts:[%d]' % len(x))

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(LightPoint.dateFormat))
        plt.gca().xaxis.set_major_locator(timeLocator())

        for i in range(len(x)):
            plt.plot(x[i], 0, k[i])

        plt.gcf().autofmt_xdate()
        plt.show()
