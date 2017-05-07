from datetime import datetime
from Device import Device
from Plot import Plot
import matplotlib.pyplot as plt


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
        # self.colors = self.colorsBars

    def setPlotDots(self):
        self.nullColor = self.nullColorDot
        self.colors = self.colorsDots

    def setPlotBars(self):
        self.nullColor = self.nullColorBar
        self.colors = self.colorsBars

    def plotDots(self, ax, k):
        for key in k.iterkeys():
            ax.plot(k[key], [0] * len(k[key]), self.colors[key], label=key)

    def plotBars(self, startDate, endDate):
        lambdaFunc = lambda x, date: date < endDate
        x, k = self.collectData(startDate, lambdaFunc, self.colorsBars)

        x.append(endDate)
        x.insert(0, startDate)
        nPts = len(x)

        print('Start date:[%s]' % x[0])
        print('End date:[%s]' % x[nPts - 1])
        print('nPts:[%d]' % nPts)

        xColorKeys = k[1]
        xColorKeys.insert(0, self.nullValue)

        xAxisLabels = []
        xAxisTicks = []
        # adding initial value to x axis
        xAxisLabels.append(x[0])
        xAxisTicks.append(0)

        fig, ax = plt.subplots()

        lastVal = 0
        legendMap = {}
        for i in range(len(x) - 1):
            val = (x[i + 1] - x[i]).seconds
            xAxisLabels.append(x[i + 1])
            xAxisTicks.append(xAxisTicks[len(xAxisTicks) - 1] + val)
            h = ax.barh(0, val, 0.2, color=self.colorsBars[xColorKeys[i]], edgecolor='grey', linewidth=0.5,
                        left=lastVal)
            legendMap[xColorKeys[i]] = h
            lastVal += val

        # building legend
        handles, labels = [], []
        for key in legendMap.iterkeys():
            handles.append(legendMap[key])
            labels.append(key)

        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Removes too close labels
        res = Plot.dateWithMinimalGap([xAxisLabels, xAxisTicks], lambda i: xAxisTicks[i] - xAxisTicks[i - 1])
        xAxisLabels = res[0]
        xAxisTicks = res[1]

        ax.set_xticks(xAxisTicks)
        ax.set_xticklabels(xAxisLabels)

        bgcolor = 0.95
        ax.set_axis_bgcolor((bgcolor, bgcolor, bgcolor))
        ax.set_title("[%s] \n Time Range: %s" % (self.filename, Plot.timedeltaToText(x[nPts - 1] - x[0])))
        plt.gcf().autofmt_xdate()
        plt.show()

    # def _Device__plotInternal(self, ax, x, k):
    #     for i in range(len(x)):
    #         ax.plot(x[i], 0, k[i])

    def _Device__plotInternal(self, ax, x, k):
        self.plotDots(ax, k[0])
        ax.legend()

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)

    def collectData(self, startDate, lambdaFunc, colors=None):
        if colors is None:
            colors = self.colors

        i = self._Device__skipToDate(startDate)

        x = []
        xColorKeys = []
        xByClass = {}

        for key in colors.iterkeys():
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
            xColorKeys.append(key)

        k = [xByClass, xColorKeys]
        return x, k
