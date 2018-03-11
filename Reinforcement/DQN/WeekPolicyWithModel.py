from abc import ABCMeta, abstractmethod
from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy


class WeekPolicyWithModel(WeekPolicy):
    __metaclass__ = ABCMeta

    def __init__(self, fname, seqLen=1):
        super(WeekPolicyWithModel, self).__init__(fname, seqLen)
        self.model = self.buildModel()

    # build model to learn policy
    @abstractmethod
    def buildModel(self):
        raise NotImplementedError('subclasses must override buildModel()!')
