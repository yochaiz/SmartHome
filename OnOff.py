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

    def _Device__plotInternal(self, ax, x, k):
        for i in range(len(x)):
            ax.plot(x[i], 0, k[i])

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)

    def collectData(self, startDate, lambdaFunc):
        i = self._Device__skipToDate(startDate)

        x = []
        k = []
        lastColor = self.nullColor

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
            if lambdaFunc(x, date) is False:
                break

            x.append(date)
            i += 1

            col = self.colors[child.text] if child.text != self.nullValue else lastColor
            k.append(col)
            lastColor = col

        return x, k
