from Device import Device
from datetime import datetime
import math


class Real(Device):
    nullTemp = 'NaN'

    def __init__(self, filename, yAxisLabel):
        Device.__init__(self, filename)
        self.yAxisLabel = yAxisLabel

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        k = []

        # update state on startDate if state exists
        iStart = i
        if iStart > 0:
            i -= 1

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

        if len(x) <= 0:
            raise ValueError('Time range has no data')

        # update state on startDate if state exists
        if iStart > 0:
            x[0] = startDate
        else:
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

        ax.set_ylabel(self.yAxisLabel)

        # # remove labels from y axis
        # nElemsY = 10
        # kUnique = sorted(list(set(k)))
        # gap = int(math.ceil(len(kUnique) / float(nElemsY)))
        # ax.set_yticks([kUnique[i] for i in range(0, len(kUnique), gap)])

        return x, xAxisTicks

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
