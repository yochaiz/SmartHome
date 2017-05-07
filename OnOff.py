from datetime import datetime
from Device import Device


class OnOff(Device):
    nullValue = "null"

    def __init__(self, filename, nullColorDot, dotColors, nullColorBar, barColors):
        super(OnOff, self).__init__(filename)

        self.nullColorDot = nullColorDot
        self.colorsDots = {self.nullValue: nullColorDot, '0': dotColors[0], '1': dotColors[1]}

        self.nullColorBar = nullColorBar
        self.colorsBars = {self.nullValue: nullColorBar, '0': barColors[0], '1': barColors[1]}

        self.nullColor = self.nullColorDot
        self.colors = self.colorsDots

    def setPlotDots(self):
        self.nullColor = self.nullColorDot
        self.colors = self.colorsDots

    def setPlotBars(self):
        self.nullColor = self.nullColorBar
        self.colors = self.colorsBars

    # def _Device__plotInternal(self, ax, x, k):
    #     for i in range(len(x)):
    #         ax.plot(x[i], 0, k[i])

    def _Device__plotInternal(self, ax, x, k):
        for key in k.iterkeys():
            ax.plot(k[key], [0] * len(k[key]), self.colors[key], label=key)

        ax.legend()

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        xByClass = {}

        for key in self.colors.iterkeys():
            xByClass[key] = []

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            x.append(date)
            i += 1

            key = child.text
            xByClass[key].append(date)


        k = xByClass
        return x, k
