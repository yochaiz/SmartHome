from Device import Device
from datetime import datetime


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
            if lambdaFunc(x, date) is False:
                break

            # we ignore null temperatures, another option is to use last valid temperature
            if child.text != self.nullTemp:
                x.append(date)
                k.append(float(child.text))

            i += 1

        return x, k

    def _Device__plotInternal(self, ax, x, k):
        ax.plot(x, k, 'bo')
        ax.set_ylabel('Temperature [Celsius]')

        return None, None

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
