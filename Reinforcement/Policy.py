from abc import ABCMeta, abstractmethod


# Abstract policy class
class Policy:
    __metaclass__ = ABCMeta

    actionValues = [0, 1]
    nActions = len(actionValues)

    def __init__(self):
        pass

    @abstractmethod
    def minTimeUnit(self):
        raise NotImplementedError('subclasses must override minTimeUnit()!')

    @abstractmethod
    def stateToDatetime(self, state):
        raise NotImplementedError('subclasses must override stateToDatetime()!')

    @abstractmethod
    def buildNextState(self, nextDate, state, action):
        raise NotImplementedError('subclasses must override buildNextState()!')


