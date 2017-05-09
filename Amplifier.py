from Device import Device
from datetime import datetime


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
        x = []
        xByClass = {}
        yByClass = {}

        for key in self.colors.iterkeys():
            xByClass[key] = []
            yByClass[key] = {}
            for j in range(1, len(self.keys)):
                yByClass[key][self.keys[j]] = []

        i = self._Device__skipToDate(startDate)
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
            # print(elem)

            if completeElement is True:
                k = self.keys[0]
                key = elem[k]
                elem[k] = self.colors[key]

                x.append(date)
                xByClass[key].append(date)
                for j in range(1, len(self.keys)):
                    yByClass[key][self.keys[j]].append(elem[self.keys[j]])

            i += 1

        y = [xByClass, yByClass]
        return x, y

    def _Device__plotInternal(self, ax, x, y):
        ax.set_ylabel("Volume [dB]")

        xByClass = y[0]
        yByClass = y[1]
        for key in xByClass.iterkeys():
            ax.__plot(xByClass[key], yByClass[key][self.keys[1]], self.colors[key], label=key)

            # annotate source value
            for i in range(len(xByClass[key])):
                ax.annotate(yByClass[key][self.keys[2]][i], (xByClass[key][i], yByClass[key][self.keys[1]][i]))

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return None, None

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)
