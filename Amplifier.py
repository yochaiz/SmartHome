from Device import Device
from datetime import datetime
import xml.etree.ElementTree as Et


class Amplifier(Device):
    keys = ['State', 'Volume', 'Source']

    nullValue = "null"
    nullColor = 'yo'
    colors = {nullValue: nullColor, '0': 'ro', '1': 'go'}

    # nullColorBar = 'yellow'
    # colorsBars = ['red', 'green']

    def __init__(self, filename):
        super(Amplifier, self).__init__(filename)

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        y = []
        lastColor = self.nullColor

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            elem = {}
            completeElement = True
            for k in self.keys:
                val = child.findall(k)
                if len(val) > 0 and val[0].text != self.nullValue:
                    elem[k] = val[0].text
                else:
                    completeElement = False
                    break
            print(elem)

            if completeElement is True:
                k = 'State'
                elem[k] = self.colors[elem[k]] if elem[k] != self.nullValue else lastColor
                lastColor = elem[k]

                x.append(date)
                y.append(elem)

            i += 1

        return x, y

    def _Device__plotInternal(self, ax, x, k):
        ax.set_ylabel("Volume")
        for i in range(len(x)):
            elem = k[i]
            ax.plot(x[i], elem['Volume'], elem['State'])
            ax.annotate(elem['Source'], (x[i], elem['Volume']))

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
