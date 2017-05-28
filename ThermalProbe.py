from Device import Device
from datetime import datetime
import math


class ThermalProbe(Device):
    nullTemp = 'NaN'

    def __init__(self, filename):
        Device.__init__(self, filename)

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        k = []

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
            if lambdaFunc(x, date) != date:
                break

            # we ignore null temperatures, another option is to use last valid temperature
            if child.text != self.nullTemp:
                x.append(date)
                k.append(float(child.text))

            i += 1

        x.insert(0, startDate)
        k.insert(0, k[0])

        return x, k

    def buildPlotData(self, x, k):
        xAxisTicks = [0]

        lastVal = 0
        for i in range(len(x) - 1):
            val = (x[i + 1] - x[i]).seconds
            xAxisTicks.append(xAxisTicks[len(xAxisTicks) - 1] + val)
            lastVal += val

        return xAxisTicks

    def _Device__plotInternal(self, ax, x, k):
        xAxisTicks = self.buildPlotData(x, k)

        # ax.plot(xAxisTicks, k, 'o', label=self.id[self.id.rfind('.') + 1:])
        ax.step(xAxisTicks, k, 'o', label=self.id[self.id.rfind('.') + 1:], where='post')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_ylabel('Temperature [Celsius]')
        nElemsY = 10
        kUnique = sorted(list(set(k)))
        gap = int(math.ceil(len(kUnique) / float(nElemsY)))
        ax.set_yticks([kUnique[i] for i in range(0, len(kUnique), gap)])

        return x, xAxisTicks

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
