from Device import Device
from datetime import datetime
from Plot import Plot


class Microphone(Device):
    keys = {'RmsVolume': 'go', 'MaxVolume': 'bo'}

    nullValue = "null"

    def __init__(self, filename):
        super(Microphone, self).__init__(filename)

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        y = []

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
                    elem[k] = float(val[0].text)
                else:
                    completeElement = False
                    break
            # print(elem)

            if completeElement is True:
                x.append(date)
                y.append(elem)

            i += 1

        # [x, y] = Plot.dateWithMinimalGap([x, y], lambda i: (x[i] - x[i - 1]).seconds)
        return x, y

    def _Device__plotInternal(self, ax, x, k):
        ax.set_ylabel("Volume [dB]")
        for i in range(len(x)):
            elem = k[i]
            for key in elem.keys():
                ax.plot(x[i], elem[key], self.keys[key])

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
