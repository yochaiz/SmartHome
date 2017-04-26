from os import listdir
from LightPoint import LightPoint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

from pylab import *


class RoomLights(object):
    dataFolder = 'data/'
    LightPointsFolder = dataFolder + 'LightPoints/'
    filePrefix = '.LightPoint.'
    fileSuffix = '.'

    def __init__(self, roomNumber):
        self.number = roomNumber
        self.LightPoints = []

        for filename in listdir(RoomLights.LightPointsFolder):
            startPos = filename.find(RoomLights.filePrefix)
            if startPos >= 0:
                startPos += len(RoomLights.filePrefix)
                endPos = filename.find(RoomLights.fileSuffix, startPos)
                if endPos >= 0:
                    rNum = filename[startPos:endPos]
                    if rNum == roomNumber:
                        self.LightPoints.append(LightPoint(RoomLights.LightPointsFolder + filename))

        print('Room [%s] contains [%d] lights' % (roomNumber, len(self.LightPoints)))

    @staticmethod
    # converts time delta object to text of the greater time type (days,hours,min,secs)
    def __timedeltaToText(t):
        if t.days > 0:
            return "[%d] Days" % t.days

        hours, remainder_ = divmod(t.seconds, 3600)
        minutes, seconds = divmod(remainder_, 60)
        if hours > 0:
            return "[%d] Hours, [%d] Minutes, [%d] Seconds" % (hours, minutes, seconds)

        if minutes > 0:
            return "[%d] Minutes, [%d] Seconds" % (minutes, seconds)

        if seconds > 0:
            return "[%d] Seconds" % seconds

        raise ValueError("Time delta is shorter than a second")

    def plot(self, startDate, endDate, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: date < endDate

        fig, ax = plt.subplots()

        counter = 1
        # padding = len(self.LightPoints) * 0.1
        # plt.ylim([(counter - padding), (len(self.LightPoints) + padding)])

        nPts = 0
        xAxisLabels = []
        xAxisTicks = []
        for lp in self.LightPoints:
            x, k = lp.collectData(startDate, lambdaFunc)
            # adding start & end dates to data
            x.append(endDate)
            x.insert(0, startDate)
            k.insert(0, lp.nullColor)
            nPts = max(nPts, len(x))
            # adding initial value to x axis
            xAxisLabels.append(x[0])
            xAxisTicks.append(0)
            lastVal = 0
            for i in range(len(x) - 1):
                val = (x[i + 1] - x[i]).seconds
                xAxisLabels.append(x[i + 1])
                xAxisTicks.append(xAxisTicks[len(xAxisTicks) - 1] + val)
                ax.barh(counter - .1, val, 0.2, color=k[i], edgecolor='grey', linewidth=0.5, left=lastVal)
                lastVal += val

            counter += 1

        ax.set_yticks(range(counter)[1:])
        ax.set_yticklabels([lp.id[lp.id.rfind(RoomLights.filePrefix) + 1:] for lp in self.LightPoints])

        # Removes too close labels
        minGap = round((xAxisLabels[len(xAxisLabels) - 1] - xAxisLabels[0]).seconds / 50.0)
        i = 1
        while i < len(xAxisTicks):
            if xAxisTicks[i] - xAxisTicks[i - 1] < minGap:
                del xAxisTicks[i]
                del xAxisLabels[i]
            else:
                i += 1

        ax.set_xticks(xAxisTicks)
        ax.set_xticklabels(xAxisLabels)

        print('nPts:[%d]' % nPts)

        bgcolor = 0.95
        ax.set_axis_bgcolor((bgcolor, bgcolor, bgcolor))
        plt.gcf().autofmt_xdate()

        timeDiff = (endDate - startDate)
        plt.title("[%s]-[%s] \n Time range: %s" % (startDate, endDate, RoomLights.__timedeltaToText(timeDiff)))
        plt.show()
