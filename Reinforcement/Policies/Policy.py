from abc import ABCMeta, abstractmethod
import numpy as np
from random import randint
import json
from datetime import datetime


# Abstract policy class
class Policy:
    __metaclass__ = ABCMeta

    dictTypes = [str, int, float, tuple]

    def __init__(self, fname, seqLen):
        self.policyJSON = self.loadPolicyFromJSON(fname)
        self.numOfDevices = len(self.policyJSON["Devices"])
        self.seqLen = seqLen

    @abstractmethod
    def minTimeUnit(self):
        raise NotImplementedError('subclasses must override minTimeUnit()!')

    # Extracts date from given state
    @abstractmethod
    def timePrefixToDate(self, state):
        raise NotImplementedError('subclasses must override stateToDatetime()!')

    # Builds state at time nextDate based on current state and selected action
    @abstractmethod
    def buildNextState(self, nextDate, state, action):
        raise NotImplementedError('subclasses must override buildNextState()!')

    @abstractmethod
    def getStateDim(self):
        raise NotImplementedError('subclasses must override getStateDim()!')

    def getActionDim(self):
        return self.numOfDevices

    # Builds the expected state at time nextDate
    def buildExpectedState(self, nextDate):
        raise NotImplementedError('subclasses must override buildExpectedState()!')

        # Builds the expected input for model of given date

    def buildDateInput(self, stateTimePrefix):
        raise NotImplementedError('subclasses must override appendExpectedState()!')

    # Calculates the reward based on the expected state at time nextDate compared to the actual state, nextState.
    def calculateReward(self, nextState, nextDate):
        raise NotImplementedError('subclasses must override calculateReward()!')

    # build model to learn policy
    def buildModel(self, logger):
        raise NotImplementedError('subclasses must override buildModel()!')

    # generate random time prefix for random state
    def generateRandomTimePrefix(self):
        raise NotImplementedError('subclasses must override generateRandomTimePrefix()!')

    # normalize state vector before it goes through the model
    def normalizeStateForModelInput(self, state):
        raise NotImplementedError('subclasses must override normalizeStateForModelInput()!')

    # generate random input for model
    def generateRandomInput(self):
        # generate random time prefix
        state = self.generateRandomTimePrefix()
        # build date object for drawn date
        date = self.timePrefixToDate(state)
        # build input for drawn date
        input = self.buildDateInput(date)

        return input

    # for exploration
    def generateRandomAction(self):
        action = np.array([], dtype=int)
        for i in range(self.numOfDevices):
            action = np.append(action, randint(0, 1))

        return action

    # perform action
    def step(self, input, action):
        curDate = self.timePrefixToDate(input)
        nextDate = curDate + self.minTimeUnit()
        nextInput = self.buildNextState(nextDate, input, action)

        # calculate reward
        reward = self.calculateReward(nextInput[-1], nextDate)

        return nextInput, reward

    def toJSON(self):
        var = dict(vars(self))  # make dict copy
        jsonObj = {}
        for key, val in var.iteritems():
            if type(val) in self.dictTypes:
                jsonObj[key] = val

        jsonObj['policyJSON'] = self.policyJSON

        return jsonObj

    # converts action vector as binary number to index (integer), for NN output index
    @staticmethod
    def actionToIdx(action):
        idx = 0
        val = pow(2, len(action) - 1)

        for i in range(len(action)):
            idx += (action[i] * val)
            val /= 2

        return idx

    # convert NN output index to corresponding action representation
    def idxToAction(self, val):
        # binary value to array of bits
        binValArray = [int(x) for x in bin(val)[2:]]

        # pad action with leading 0's
        action = []
        if len(binValArray) < self.numOfDevices:
            action = [0] * (self.numOfDevices - len(binValArray))

        # merge leading 0's with binary bits
        action.extend(binValArray)
        action = np.array(action, dtype=int)

        return action

    # prints model architecture
    @staticmethod
    def printModel(model, logger):
        model.summary(print_fn=lambda x: logger.info(x))

    @staticmethod
    def loadPolicyFromJSON(fname):
        with open(fname, 'r') as f:
            policy = json.load(f)

        Policy.validatePolicy(policy)
        return policy

    @staticmethod
    def validatePolicy(policy):
        nDays = len(policy["days"])
        for i in range(len(policy["Devices"])):
            device = policy[str(i)]
            daysArray = [[] for j in range(nDays)]

            for timeDict in device:
                if type(timeDict[
                            "days"]) is unicode:  # replace predefined array in JSON with actual array for future simplicity
                    timeDict["days"] = policy[timeDict["days"]]

                # sort timeDict by startTime
                timeDict["times"] = sorted(timeDict["times"],
                                           key=lambda t: datetime.strptime(t[0], policy["Time format"]))

                for day in timeDict["days"]:
                    daysArray[day].extend(timeDict["times"])

            # sort time ranges for easier compare
            for j in range(len(daysArray)):
                daysArray[j] = sorted(daysArray[j], key=lambda t: datetime.strptime(t[0], policy["Time format"]))

            for array in daysArray:
                for j in range(len(array) - 1):
                    tCur = array[j]
                    tNext = array[j + 1]
                    if tCur[1] > tNext[0]:  # endTime is bigger than next range startTime
                        raise ValueError(
                            'Validation failed for device [{}], ID:[{}], time ranges: [{}] - [{}]'.format(
                                policy["Devices"][i], i, tCur, tNext))
