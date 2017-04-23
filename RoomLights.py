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

    def plot(self, startDate, endDate, timeLocator=mdates.HourLocator):
        lambdaFunc = lambda x, date: date < endDate

        fig, ax = plt.subplots()
        val = [9, 1, 2, 3, 4]  # the bar lengths
        pos = range(5)  # the bar centers on the y axis
        ax.barh(pos, val, 0.2, align='center')
        ax.set_yticks(pos)
        ax.set_yticklabels(('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
        ax.invert_yaxis()
        ax.set_xticklabels(range(120, 130))

        dd = endDate - startDate
        step = timedelta(seconds=round(dd.seconds / 10.0))
        val = startDate
        rr = []
        while val < endDate:
            rr.append(val)
            val += step

        ax.set_xticklabels(rr)
        plt.gcf().autofmt_xdate()

        xlabel('Performance')
        show()

        # fig, ax = plt.subplots()
        #
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(LightPoint.dateFormat))
        # plt.gca().xaxis.set_major_locator(timeLocator())
        #
        # counter = 1
        # # padding = len(self.LightPoints) * 0.1
        # # plt.ylim([(counter - padding), (len(self.LightPoints) + padding)])
        #
        # nPts = 0
        # for lp in self.LightPoints:
        #     x, k = lp.collectData(startDate, lambdaFunc)
        #     nPts = max(nPts, len(x))
        #     lastVal = 0
        #     for i in range(len(x) - 1):
        #         # plt.plot(x[i], counter, k[i])
        #         val = x[i + 1] - x[i]
        #         val = val.seconds / 3600.0
        #         plt.barh(counter - .1, val, 0.2, color=k[i], left=lastVal)
        #         lastVal += val
        #
        #     counter += 1
        #
        # ax.set_yticks(range(counter)[1:])
        # ax.set_yticklabels([lp.id[lp.id.rfind(RoomLights.filePrefix) + 1:] for lp in self.LightPoints])
        #
        # # step = x[len(x) - 1] - x[0]
        # # step = timedelta(seconds=round(step.seconds / 10.0))
        # # dateRange = []
        # # val = x[0]
        # # while val < x[len(x) - 1]:
        # #     dateRange.append(val)
        # #     val += step
        # #
        # # ax.set_xticks(dateRange)
        #
        # print('nPts:[%d]' % nPts)
        #
        # plt.gcf().autofmt_xdate()
        # plt.show()
