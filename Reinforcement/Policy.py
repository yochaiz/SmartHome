from abc import ABCMeta, abstractmethod
import numpy as np
from random import randint


# Abstract policy class
class Policy:
    __metaclass__ = ABCMeta

    def __init__(self, fname, seqLen):
        self.policyJSON = self.loadPolicyFromJSON(fname)
        self.numOfDevices = len(self.policyJSON["Devices"])
        self.seqLen = seqLen
        self.model = self.buildModel()

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
        return self.policyJSON

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
