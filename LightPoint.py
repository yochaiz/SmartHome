import xml.etree.ElementTree as Et
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Device import Device


class LightPoint(Device):
    # dateFormat = '%Y-%m-%d %H:%M:%S.%f'
    # dateFormat = '%Y-%m-%d %H:%M:%S'
    nullColor = 'red'
    nullValue = "null"
    colors = {nullValue: nullColor, '0': 'black', '1': 'yellow'}

    # colors = [nullColor, 'black', 'yellow']

    # nullColor = 'ro'
    # colors = [nullColor, 'ko', 'yo']

    def __init__(self, filename):
        Device.__init__(self, filename)

    def _Device__plot(self, startDate, lambdaFunc, timeLocator):
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
        i = self._Device__skipToDate(startDate)

        x = []
        k = []
        lastColor = LightPoint.nullColor

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], LightPoint.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            x.append(date)
            i += 1

            col = LightPoint.colors[child.text] if child.text != LightPoint.nullValue else lastColor
            k.append(col)
            lastColor = col

        return x, k
