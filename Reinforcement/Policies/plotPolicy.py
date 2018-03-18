from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
# from os import sys, path
#
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from WeekPolicy import WeekPolicy


# Given a device times dictionary and a specific day, collect all the time ranges of the specific day
# time ranges are returned SORTED
def collectDeviceSingleDayData(policy, device, day, timeFormat):
    data = []

    for timeDict in device:
        if type(timeDict["days"]) is list:
            days = timeDict["days"]
        elif type(timeDict["days"]) is unicode:  # predefined array in JSON
            days = policy[timeDict["days"]]
        else:
            raise ValueError("Unknown days key value")

        if day in days:
            data.extend(timeDict["times"])

        data = sorted(data, key=lambda t: datetime.strptime(t[0], timeFormat))

    return data


def plotDataForDeviceSingleDay(policy, device, day, timeFormat):
    timeData = collectDeviceSingleDayData(policy, device, day, timeFormat)

    curTime = datetime.strptime("00:00", policy["Time format"])
    lastVal = 0
    xAxisLabels = []
    width = []
    color = []
    xAxisTicks = []
    for t in timeData:
        startTime = datetime.strptime(t[0], policy["Time format"])
        endTime = datetime.strptime(t[1], policy["Time format"])

        if curTime < startTime:
            timeOn = int((startTime - curTime).total_seconds())
            width.append(timeOn)
            xAxisTicks.append(lastVal)
            lastVal += timeOn
            color.append('k')  # device is off
            xAxisLabels.append(curTime.strftime(policy["Time format"]))

        timeOn = int((endTime - startTime + timedelta(minutes=1)).total_seconds())
        width.append(timeOn)
        xAxisTicks.append(lastVal)
        xAxisLabels.append(t[0])

        lastVal += timeOn
        color.append('y')  # device is on

        curTime = endTime + timedelta(minutes=1)

    # complete missing values for a full day plot
    xAxisLabels.append(curTime.strftime(policy["Time format"]))
    endTime = datetime.strptime("23:59", policy["Time format"]) + timedelta(minutes=1)
    if curTime < endTime:
        xAxisLabels.append(endTime.strftime(policy["Time format"]))
        xAxisTicks.append(lastVal)

        timeOn = int((endTime - curTime + timedelta(minutes=1)).total_seconds())
        lastVal += timeOn
        width.append(timeOn)
        color.append('k')  # device is off

    # plot
    height = 0.2

    xAxisTicks.append(lastVal)  # leftValues is xAxisTicks WITHOUT last element
    plotData = [width, height, color, xAxisTicks, xAxisLabels]

    return plotData


def plotDataForDeviceMultipleDays(policy, device, timeFormat, days):
    # 1st day does not need shift, just get values
    plotData = plotDataForDeviceSingleDay(policy, device, days[0], timeFormat)
    # following days require shift
    for dayID in range(1, len(days)):
        dayPlotData = plotDataForDeviceSingleDay(policy, device, days[dayID], timeFormat)

        # width and color array do not need shift
        for i in [0, 2]:
            plotData[i].extend(dayPlotData[i])

        #  xAxisTicks array need shift
        for i in [3]:
            shiftValue = plotData[i][-1]
            plotData[i].extend([(v + shiftValue) for v in dayPlotData[i][1:]])

        # xAxisLabels array does not need shift but need to remove overlapping values
        plotData[4].extend(dayPlotData[4][1:])

    return plotData


# list1 is array of arrays: [ticks, labels]
# list2 is the same
def mergeSortXaxis(list1, list2):
    i, j = 0, 0
    mergedTicks, mergedLabels = [None], []
    ticks1, ticks2 = list1[0], list2[0]

    while (i < len(ticks1)) and (j < len(ticks2)):
        if ticks1[i] <= ticks2[j]:
            if ticks1[i] != mergedTicks[-1]:
                mergedTicks.append(ticks1[i])
                mergedLabels.append(list1[1][i])
            i += 1

        elif ticks2[j] < ticks1[i]:
            if ticks2[j] != mergedTicks[-1]:
                mergedTicks.append(ticks2[j])
                mergedLabels.append(list2[1][j])
            j += 1

    while i < len(ticks1):
        if ticks1[i] != mergedTicks[-1]:
            mergedTicks.append(ticks1[i])
            mergedLabels.append(list1[1][i])
        i += 1

    while j < len(ticks2):
        if ticks2[j] != mergedTicks[-1]:
            mergedTicks.append(ticks2[j])
            mergedLabels.append(list2[1][j])
        j += 1

    mergedTicks = mergedTicks[1:]  # remove dummy object
    return mergedTicks, mergedLabels


# remove values that are too close
def spaceXaxis(xTicks, xLabels):
    minDiff = 300  # value in seconds, i.e. 5 minutes
    i = 1
    while i < len(xTicks):
        diff = xTicks[i] - xTicks[i - 1]
        if diff < minDiff:
            del xTicks[i]
            del xLabels[i]
        else:
            i += 1

    return xTicks, xLabels


# policy - policy as JSON
# days - array of days as integers
def plotPolicy(policy, days):
    fig, ax = plt.subplots()
    fig.autofmt_xdate()

    if type(days) is not list:  # predefined days array in JSON
        days = policy[days]

    yLabels, yTicks = [], []
    xLabels, xTicks = [], []

    for deviceID, name in enumerate(policy["Devices"]):
        device = policy[str(deviceID)]
        plotData = plotDataForDeviceMultipleDays(policy, device, policy["Time format"], days)

        ax.barh([deviceID] * len(plotData[0]), plotData[0], height=plotData[1], color=plotData[2], left=plotData[3][:-1])

        yLabels.append(name)
        yTicks.append(deviceID + (plotData[1] / 2))
        xTicks, xLabels = mergeSortXaxis([xTicks, xLabels], [plotData[3], plotData[4]])

    xTicks, xLabels = spaceXaxis(xTicks, xLabels)
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabels)
    ax.set_yticks(yTicks)
    ax.set_yticklabels(yLabels)
    ax.set_title('Devices:[All] - Days:{}'.format(days))

    return ax


# parse arguments
def parseArguments():
    parser = argparse.ArgumentParser(description='Plot policy')
    parser.add_argument("policyFile", type=str, help="Policy JSON file path")
    parser.add_argument("days", type=int, choices=range(-3, 7),
                        help="Days of policy to plot: 0-Monday, 1-Tuesday, 2-Wednesday, 3-Thursday, 4-Friday, 5-Saturday, 6-Sunday, (-1)-full week, (-2)-weekdays, (-3)-weekend")

    args = parser.parse_args()
    # convert args.days
    if args.days == -1:
        args.days = range(7)
    elif args.days >= 0:
        args.days = [args.days]
    elif args.days == -2:
        args.days = "weekdays"
    else:
        args.days = "weekend"

    return args


args = parseArguments()
policy = WeekPolicy(args.policyFile)

ax = plotPolicy(policy.policyJSON, args.days)

plt.show()
