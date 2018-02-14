import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def loadPolicy(fname):
    with open(fname, 'r') as f:
        policy = json.load(f)

    return policy


# Given a device times dictionary and a specific day, collect all the time ranges of the specific day
# time ranges are returned SORTED
def collectDeviceSingleDayData(device, day, timeFormat):
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


def plotDeviceSingleDay(device, day, timeFormat):
    timeData = collectDeviceSingleDayData(device, day, timeFormat)

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
            timeOn = (startTime - curTime).seconds
            width.append(timeOn)
            xAxisTicks.append(lastVal)
            lastVal += timeOn
            color.append('k')  # device is off
            xAxisLabels.append(curTime.strftime(policy["Time format"]))

        timeOn = (endTime - startTime + timedelta(minutes=1)).seconds
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

        timeOn = (endTime - curTime + timedelta(minutes=1)).seconds
        lastVal += timeOn
        width.append(timeOn)
        color.append('k')  # device is off

    # plot
    height = 0.2

    xAxisTicks.append(lastVal)  # leftValues is xAxisTicks WITHOUT last element
    plotData = [width, height, color, xAxisTicks, xAxisLabels]

    return plotData


def plotDeviceMultipleDays(device, timeFormat, days):
    # 1st day does not need shift, just get values
    plotData = plotDeviceSingleDay(device, days[0], timeFormat)
    # following days require shift
    for dayID in range(1, len(days)):
        dayPlotData = plotDeviceSingleDay(device, days[dayID], timeFormat)

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


# TODO: validate json, no overlap between time intervals

policyFilename = '../Week_policies/policy1.json'
policy = loadPolicy(policyFilename)

deviceID = 0
dayID = 4
device = policy[str(deviceID)]
# plotData = plotDeviceSingleDay(device, 0, policy["Time format"])
# plotData1 = plotDeviceSingleDay(device, 1, policy["Time format"])
# plotData1[0] = [(v + plotData[0][-1]) for v in plotData1[0]]
plotData = plotDeviceMultipleDays(device, policy["Time format"], [6, 0, 1, 2, 3, 4, 5])

fig, ax = plt.subplots()
fig.autofmt_xdate()

ax.barh([0.5] * len(plotData[0]), plotData[0], height=plotData[1], color=plotData[2], left=plotData[3][:-1])
ax.set_xticks(plotData[3])
ax.set_xticklabels(plotData[4])
ax.set_title('Device:[{}]'.format(policy["Devices"][deviceID]))
plt.show()
