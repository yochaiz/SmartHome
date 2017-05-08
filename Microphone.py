from Device import Device
from datetime import datetime
from Plot import Plot


class Microphone(Device):
    keys = {'RmsVolume': 'go', 'MaxVolume': 'bo'}

    nullValue = "null"

    def __init__(self, filename):
        super(Microphone, self).__init__(filename)

    def collectData(self, startDate, lambdaFunc):
        x = []
        xByClass = {}
        yByClass = {}
        for key in self.keys.iterkeys():
            xByClass[key] = []
            yByClass[key] = []

        i = self._Device__skipToDate(startDate)
        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            elem = {}
            for k in self.keys:
                val = child.findall(k)
                if len(val) > 0 and val[0].text != self.nullValue:
                    elem[k] = float(val[0].text)
            # print(elem)

            if len(elem.keys()) > 0:
                x.append(date)
                for key in elem.keys():
                    xByClass[key].append(date)
                    yByClass[key].append(elem[key])

            i += 1

        return x, [xByClass, yByClass]

    def _Device__plotInternal(self, ax, x, k):
        ax.set_ylabel("Volume [dB]")

        xByClass = k[0]
        yByClass = k[1]

        for key in xByClass.keys():
            ax.plot(xByClass[key], yByClass[key], self.keys[key], label=key)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return None, None

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
