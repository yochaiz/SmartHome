import os
from datetime import datetime, timedelta
import logging
import numpy as np
from XmlFile import XmlFile
import h5py
from ExperimentLogger import ExperimentLogger


class DatasetBuilder(object):
    folders = {'AuxChannel', 'LightPoints'}
    dateFormat = '%Y-%m-%d %H:%M:%S'
    batchSize = 32
    xDataFile = h5py.File('x-1-second.h5', 'w')
    yDataFile = h5py.File('y-1-second.h5', 'w')

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

        startDate = startDate.replace(microsecond=0)
        endDate = endDate.replace(microsecond=0)
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
        state = np.array([date.weekday(), date.hour, date.minute, date.second])
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

    def build(self, folderPath, seqLen):
        self.headers = ['Day of week', 'Hour', 'Minute', 'Second']
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
        interval = timedelta(seconds=1)
        lastDate += interval  # last date in XMLs
        endDate = date + interval  # interval end date
        logger.info('Start date:[%s] - [%s], lastDate:[%s]' % (date, date.weekday(), lastDate))

        import time
        # t1 = time.mktime(date.timetuple())
        # t2 = time.mktime(lastDate.timetuple())
        # nSamples = int(t2 - t1) / 60
        nSamples = 24000000
        logger.info('nSamples:[%d]' % nSamples)

        xSet = self.xDataFile.create_dataset("default", (nSamples, seqLen, len(self.headers)))
        ySet = self.yDataFile.create_dataset("default", (nSamples, 11))

        # collect data
        sampleIdx = 0
        # x is equivalent to sentence, it is a seqLen length sentence built from words (states)
        x = np.ndarray(shape=(seqLen, len(self.headers)))
        for i in range(seqLen):
            # state is equivalent to word in sentence, it is its vector representation
            state = self.buildState(date, endDate)
            x[i, :] = state
            date = endDate
            endDate += interval

        lightPointsStartIdx = 7
        y = self.buildState(date, endDate)
        xSet[sampleIdx, :, :] = x
        ySet[sampleIdx, :] = y[0, lightPointsStartIdx:]  # output should predict the probability for each LightPoint to be turned on

        logger.info(xSet[sampleIdx, :, :])
        logger.info(ySet[sampleIdx, :])

        sampleIdx += 1

        while (sampleIdx < nSamples) and (endDate <= lastDate):
            x = x[1:, :]  # remove oldest state from sequence
            x = np.append(x, y, axis=0)  # add new state to sequence
            date = endDate
            endDate += interval

            y = self.buildState(date, endDate)
            xSet[sampleIdx, :, :] = x
            ySet[sampleIdx, :] = y[0, lightPointsStartIdx:]  # output should predict the probability for each LightPoint to be turned on

            sampleIdx += 1
            if sampleIdx % 10000 == 0:
                logger.info(sampleIdx)

            logger.info('Done !')

logger = ExperimentLogger().getLogger()
b = DatasetBuilder()
# logging.basicConfig(level=logging.DEBUG)
b.build('../data/LivingRoom', 10)
