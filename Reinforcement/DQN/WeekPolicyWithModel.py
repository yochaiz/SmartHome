from abc import ABCMeta, abstractmethod
from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy
from keras.models import clone_model


class WeekPolicyWithModel(WeekPolicy):
    __metaclass__ = ABCMeta

    mainModelKey = 'main'
    targetModelKey = 'target'

    def __init__(self, fname, TAU, seqLen=1):
        super(WeekPolicyWithModel, self).__init__(fname, seqLen)

        self.TAU = TAU

        # create models
        self.models = {}
        # create main model
        self.models[self.mainModelKey] = self.buildModel()
        # create target (final) model as copy of training model
        self.models[self.targetModelKey] = clone_model(self.models[self.mainModelKey])
        self.models[self.targetModelKey].set_weights(self.models[self.mainModelKey].get_weights())

        # build model to learn policy

    @abstractmethod
    def buildModel(self):
        raise NotImplementedError('subclasses must override buildModel()!')

    def getMainModel(self):
        return self.models[self.mainModelKey]

    def getTargetModel(self):
        return self.models[self.targetModelKey]

    # update target model parameters SLOWLY by current trained model parameters
    def updateModelParams(self):
        wModel = self.models[self.mainModelKey].get_weights()
        wTargetModel = self.models[self.targetModelKey].get_weights()
        assert (len(wModel) == len(wTargetModel))
        for i in xrange(len(wTargetModel)):
            wTargetModel[i] = (self.TAU * wModel[i]) + ((1 - self.TAU) * wTargetModel[i])
