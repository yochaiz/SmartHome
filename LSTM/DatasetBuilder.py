import os
from datetime import datetime, timedelta
import logging
import numpy as np
from XmlFile import XmlFile
import h5py
import argparse
from ExperimentLogger import ExperimentLogger
from abc import ABCMeta, abstractmethod


class TimeResolution(object):
    __metaclass__ = ABCMeta

    def __init__(self, val):
        self.value = val

    @abstractmethod
    def resetDate(self, date):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))

    @abstractmethod
    def statePrefix(self, date):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))

    @abstractmethod
    def headerPrefix(self):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))

    @abstractmethod
    def interval(self):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))

    @abstractmethod
    def toString(self):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))

    @abstractmethod
    def lightPointStartIdx(self):
        raise NotImplementedError('subclasses must override [{}]'.format(__name__))


class MinuteResolution(TimeResolution):
    def __init__(self, val):
        super(MinuteResolution, self).__init__(val)

    def resetDate(self, date):
        return date.replace(second=0, microsecond=0)

    def statePrefix(self, date):
        return np.array([date.weekday(), date.hour, date.minute])

    def headerPrefix(self):
        return ['Day of week', 'Hour', 'Minute']

    def interval(self):
        return timedelta(minutes=self.value)

    def toString(self):
        return '{}-minute'.format(self.value)

    def lightPointStartIdx(self):
        return 6


class SecondResolution(TimeResolution):
    def __init__(self, val):
        super(SecondResolution, self).__init__(val)

    def resetDate(self, date):
        return date.replace(microsecond=0)

    def statePrefix(self, date):
        return np.array([date.weekday(), date.hour, date.minute, date.second])

    def headerPrefix(self):
        return ['Day of week', 'Hour', 'Minute', 'Second']

    def interval(self):
        return timedelta(seconds=self.value)

    def toString(self):
        return '{}-second'.format(self.value)

    def lightPointStartIdx(self):
        return 7


class DatasetBuilder(object):
    folders = {'AuxChannel', 'LightPoints'}
    dateFormat = '%Y-%m-%d %H:%M:%S'

    # batchSize = 1024

    def __init__(self, args):
        if args.second is not None:
            assert (args.minute is None)
            self.timeRes = SecondResolution(args.second)
        else:
            assert (args.second is None)
            self.timeRes = MinuteResolution(args.minute)

        self.parentFolderName = args.dstFolder
        self.folderName = '{}/{}-seqLen-{}'.format(self.parentFolderName, self.timeRes.toString(), args.seqLen)
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)
        else:
            print('Folder [{}] already exists, please remove it and run again'.format(self.folderName))
            exit(0)

        self.seqLen = args.seqLen
        self.nSamples = args.nSamples
        self.nSamplesPerFile = args.nSamplesPerFile

        self.logger = ExperimentLogger(self.folderName).getBasicLogger()

    def updateFilesPosition(self, startDate):
        for deviceFolder in self.data:
            for device in self.data[deviceFolder]:
                child = device.xml[device.pos]
                date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
                while date < startDate:
                    device.pos += 1
                    child = device.xml[device.pos]
                    date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)

    def getInitialDateRange(self):
        for deviceFolder in self.data:
            if len(self.data[deviceFolder]) > 0:
                startChild = self.data[deviceFolder][0].xml[0]
                endChild = self.data[deviceFolder][0].xml[-1]
                break

        startDate = datetime.strptime(startChild.get('Time')[:-3], self.dateFormat)
        endDate = datetime.strptime(endChild.get('Time')[:-3], self.dateFormat)

        for deviceFolder in self.data:
            for device in self.data[deviceFolder]:
                child = device.xml[0]
                date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
                logging.debug('filename:[%s] - date:[%s] - startDate:[%s]' % (device.xml.get('Id'), date, startDate))
                if date > startDate:
                    startDate = date

                child = device.xml[-1]
                date = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
                logging.debug('filename:[%s] - date:[%s] - endDate:[%s]' % (device.xml.get('Id'), date, endDate))
                if date < endDate:
                    endDate = date

        startDate = self.timeRes.resetDate(startDate)
        endDate = self.timeRes.resetDate(endDate)
        # startDate = startDate.replace(microsecond=0)
        # endDate = endDate.replace(microsecond=0)
        # startDate = startDate.replace(second=0, microsecond=0)
        # endDate = endDate.replace(second=0, microsecond=0)

        # update positions for all files
        self.updateFilesPosition(startDate)

        return startDate, endDate

    def collectFiles(self, folderPath, folderName):
        self.data[folderName] = []
        for filename in os.listdir(folderPath + '/' + folderName):
            self.data[folderName].append(XmlFile(folderPath + '/' + folderName + '/' + filename))
            self.headers.append(filename)

    def buildState(self, date, endDate):
        state = self.timeRes.statePrefix(date)
        # state = np.array([date.weekday(), date.hour, date.minute, date.second])
        for deviceFolder in self.data:
            for device in self.data[deviceFolder]:
                histogram = np.array([0, 0])  # counts device states in given interval [date,endDate]
                child = device.xml[device.pos]
                deviceDate = datetime.strptime(child.get('Time')[:-3], self.dateFormat)
                while deviceDate < endDate:
                    logging.debug('Date:[%s] - Value:[%s]' % (deviceDate, child.text))
                    if child.text != 'null':
                        histogram[int(child.text)] += 1
                    device.pos += 1
                    child = device.xml[device.pos]
                    deviceDate = datetime.strptime(child.get('Time')[:-3], self.dateFormat)

                if np.max(histogram) > 0:  # in case there are records for this interval that aren't NULLs
                    device.lastVal = np.argmax(histogram)

                state = np.append(state, device.lastVal)
                logging.debug('Most frequent value:[%d]' % device.lastVal)

        state = state.reshape((1, state.shape[0]))
        return state  # returns row vector

    def build(self, folderPath):
        self.headers = self.timeRes.headerPrefix()
        # self.headers = ['Day of week', 'Hour', 'Minute', 'Second']
        self.data = {}
        for f in self.folders:
            self.collectFiles(folderPath, f)

        noFilesFlag = True
        for deviceFolder in self.data:
            if len(self.data[deviceFolder]) > 0:
                noFilesFlag = False

        if noFilesFlag is True:
            return

        # set initial start date, 0 is Monday
        date, lastDate = self.getInitialDateRange()
        interval = self.timeRes.interval()
        # interval = timedelta(seconds=1)
        lastDate += interval  # last date in XMLs
        endDate = date + interval  # interval end date
        self.logger.info('Start date:[%s] - [%s], lastDate:[%s]' % (date, date.weekday(), lastDate))

        import time
        # t1 = time.mktime(date.timetuple())
        # t2 = time.mktime(lastDate.timetuple())
        # nSamples = int(t2 - t1) / 60
        # self.nSamples = 24000000
        self.logger.info('nSamples:[%d]' % self.nSamples)

        fileID = 1
        xDataFile = h5py.File('{}/x-{}.h5'.format(self.folderName, fileID), 'w')
        yDataFile = h5py.File('{}/y-{}.h5'.format(self.folderName, fileID), 'w')
        xSet = xDataFile.create_dataset("default", (self.nSamplesPerFile, self.seqLen, len(self.headers)))
        ySet = yDataFile.create_dataset("default", (self.nSamplesPerFile, 11))
        fileID += 1

        # collect data
        fileSampleIdx = 0
        totalSamplesCounter = 0
        # x is equivalent to sentence, it is a seqLen length sentence built from words (states)
        x = np.ndarray(shape=(self.seqLen, len(self.headers)))
        for i in range(self.seqLen):
            # state is equivalent to word in sentence, it is its vector representation
            state = self.buildState(date, endDate)
            x[i, :] = state
            date = endDate
            endDate += interval

        lightPointsStartIdx = self.timeRes.lightPointStartIdx()
        y = self.buildState(date, endDate)
        xSet[fileSampleIdx, :, :] = x
        ySet[fileSampleIdx, :] = y[0, lightPointsStartIdx:]  # output should predict the probability for each LightPoint to be turned on

        self.logger.info(xSet[fileSampleIdx, :, :])
        self.logger.info(ySet[fileSampleIdx, :])

        fileSampleIdx += 1
        totalSamplesCounter += 1

        while (totalSamplesCounter < self.nSamples) and (endDate <= lastDate):
            print('totalSamplesCounter:[{}] - fileSampleIdx:[{}]'.format(totalSamplesCounter, fileSampleIdx))
            if fileSampleIdx == self.nSamplesPerFile:
                fileSampleIdx = 0

                xDataFile.close()
                yDataFile.close()
                xDataFile = h5py.File('{}/x-{}.h5'.format(self.folderName, fileID), 'w')
                yDataFile = h5py.File('{}/y-{}.h5'.format(self.folderName, fileID), 'w')
                xSet = xDataFile.create_dataset("default", (self.nSamplesPerFile, self.seqLen, len(self.headers)))
                ySet = yDataFile.create_dataset("default", (self.nSamplesPerFile, 11))
                fileID += 1

            x = x[1:, :]  # remove oldest state from sequence
            x = np.append(x, y, axis=0)  # add new state to sequence
            date = endDate
            endDate += interval

            y = self.buildState(date, endDate)
            xSet[fileSampleIdx, :, :] = x
            ySet[fileSampleIdx, :] = y[0, lightPointsStartIdx:]  # output should predict the probability for each LightPoint to be turned on

            fileSampleIdx += 1
            totalSamplesCounter += 1
            if totalSamplesCounter % 10000 == 0:
                self.logger.info(totalSamplesCounter)

        self.logger.info('Done !')


parser = argparse.ArgumentParser(description='Build dataset based on arguments.')
parser.add_argument("seqLen", type=int, choices=range(1, 1001), metavar='seqLen:[1-1,000]', help="Sample sequence length")
parser.add_argument("nSamples", type=int, choices=range(1, 100000001), metavar='nSamples:[1-100,000,000]', help="Number of samples in dataset")
parser.add_argument("--nSamplesPerFile", type=int, default=50000, choices=range(1, 1000001), metavar='[1-1,000,000]', help="Number of samples per dataset file")
parser.add_argument("--dstFolder", type=str, default='datasets', help="Dataset parent folder location")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--second", type=int, choices=range(1, 1000001), metavar='[1-1,000,000]', help="Samples time resolution is 1 second")
group.add_argument("--minute", type=int, choices=range(1, 1000001), metavar='[1-1,000,000]', help="Samples time resolution is 1 minute")
args = parser.parse_args()

b = DatasetBuilder(args)
b.build('../data/LivingRoom')
