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

        self.nullColor = self.nullColorBar
        self.colors = self.colorsBars
        # self.colors = self.colorsBars

    def setPlotDots(self):
        self.nullColor = self.nullColorDot
        self.colors = self.colorsDots

    def setPlotBars(self):
        self.nullColor = self.nullColorBar
        self.colors = self.colorsBars

    # def __mergeXaxis(self, axTicks, axLabels, xAxisTicks, xAxisLabels):
    #     i = 0
    #     j = 0
    #     while i < len(axTicks):
    #         while j < len(xAxisTicks) and axTicks[i] > xAxisTicks[j]:
    #             j += 1
    #
    #         if j >= len(xAxisTicks) or axTicks[i] < xAxisTicks[j]:
    #             xAxisTicks.insert(j, axTicks[i])
    #             xAxisLabels.insert(j, datetime.strptime(axLabels[i]._text, self.dateFormat))
    #             # xAxisLabels.insert(j, str(axLabels[i]._text))
    #
    #         i += 1
    #
    # def __sortPlotXaxis(self, ax, x, xAxisLabels, xAxisTicks):
    #     self._Device__sortPlotXaxis(ax, x, xAxisLabels, xAxisTicks)
    #
    # def _Device__sortPlotXaxis(self, ax, x, xAxisLabels, xAxisTicks):
    #     # merge labels
    #     self.__mergeXaxis(ax.get_xticks(), ax.get_xticklabels(), xAxisTicks, xAxisLabels)
    #
    #     # Removes too close labels
    #     minGap = round((xAxisTicks[len(xAxisTicks) - 1] - xAxisTicks[0]) / 50.0)
    #     res = Plot.dateWithMinimalGap([xAxisLabels, xAxisTicks], lambda i: xAxisTicks[i] - xAxisTicks[i - 1], minGap)
    #     xAxisLabels = res[0]
    #     xAxisTicks = res[1]
    #
    #     ax.set_xticks(xAxisTicks)
    #     ax.set_xticklabels(xAxisLabels)
    #
    #     bgcolor = 0.95
    #     ax.set_axis_bgcolor((bgcolor, bgcolor, bgcolor))

    def buildPlotData(self, x, k):
        xColorKeys = k[1]
        # xColorKeys.insert(0, self.nullValue)

        xAxisLabels = []
        xAxisTicks = []
        # adding initial value to x axis
        xAxisLabels.append(x[0])
        xAxisTicks.append(0)

        lastVal = 0
        legendMap = {}
        width = []
        color = []
        leftValues = []
        for i in range(len(x) - 1):
            val = (x[i + 1] - x[i]).seconds
            width.append(val)
            color.append(self.colorsBars[xColorKeys[i]])
            leftValues.append(lastVal)
            # xAxisLabels.append(x[i + 1])
            xAxisLabels.append(x[i + 1])
            xAxisTicks.append(xAxisTicks[len(xAxisTicks) - 1] + val)
            # h = ax.barh(yPos, val, 0.2, color=self.colorsBars[xColorKeys[i]], edgecolor='grey', linewidth=0.5,
            #             left=lastVal)
            legendMap[xColorKeys[i]] = i
            lastVal += val

        height = 0.2
        plotData = [width, height, color, leftValues, legendMap]

        return plotData, xAxisLabels, xAxisTicks

    def _Device__plotInternal(self, ax, x, k):
        plotData, xAxisLabels, xAxisTicks = self.buildPlotData(x, k)

        yticks = ax.get_yticks().tolist()
        yticklabels = ax.get_yticklabels()
        pos = len(yticks)
        yticks.append(pos)
        pos -= (plotData[1] / 2.0)
        yticklabels.append(self.id)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        h = ax.barh([pos] * len(plotData[0]), plotData[0], height=plotData[1], color=plotData[2], left=plotData[3],
                    edgecolor='grey', linewidth=0.5)

        # building legend
        legendMap = plotData[4]
        legend = ax.get_legend()
        labels = [] if legend is None else [str(x._text) for x in legend.texts]
        handles = [] if legend is None else legend.legendHandles
        # handles, labels = ax.get_legend_handles_labels()
        for key in legendMap.iterkeys():
            if key not in labels:
                handles.append(h[legendMap[key]])
                labels.append(key)

        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        h1,l1 = ax.get_legend_handles_labels()
        # self.__sortPlotXaxis(ax, x, xAxisLabels, xAxisTicks)

        return xAxisLabels, xAxisTicks

    def __plotInternal(self, ax, x, k):
        self._Device__plotInternal(ax, x, k)

    def collectData(self, startDate, lambdaFunc):
        colors = self.colors

        i = self._Device__skipToDate(startDate)

        x = []
        xColorKeys = []
        xByClass = {}

        for key in colors.iterkeys():
            xByClass[key] = []

        x.append(startDate)
        xColorKeys.append(self.nullValue)
        xByClass[self.nullValue].append(startDate)

        while i < len(self.root):
            child = self.root[i]
            date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)

            date = lambdaFunc(x, date)
            if date is None:
                break

            x.append(date)
            i += 1

            key = child.text
            xByClass[key].append(date)
            xColorKeys.append(key)

        k = [xByClass, xColorKeys]
        return x, k

# def plotDots(self, ax, k):
#     for key in k.iterkeys():
#         ax.plot(k[key], [0] * len(k[key]), self.colors[key], label=key)

# def _Device__plotInternal(self, ax, x, k):
#     self.plotDots(ax, k[0])
#     ax.legend()

# def buildBarsPlot(self, ax, startDate, endDate, yHeight=0):
#     lambdaFunc = lambda x, date: date < endDate
#     x, k = self.collectData(startDate, lambdaFunc, self.colorsBars)

# x.append(endDate)
# x.insert(0, startDate)
# nPts = len(x)

# print('Start date:[%s]' % x[0])
# print('End date:[%s]' % x[nPts - 1])
# print('nPts:[%d]' % nPts)
