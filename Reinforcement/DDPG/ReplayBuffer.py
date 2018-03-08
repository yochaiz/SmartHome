from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, size, gamma):
        self.memory = deque(maxlen=size)
        self.gamma = gamma

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # def replay(self, actorTrainModel, actorTargetModel, actorTrainFunc, actorUpdateEpsilonFunc, criticTrainModel, criticTargetModel,
    #            criticGradientsFunc,
    #            policyNormalizeStateForModelInput, updateTargetModelsParams, trainSetSize):
    def replay(self, actor, critic, policy, updateTargetModelsParams, trainSetSize):
        # Sample trainSet from the memory
        trainSet = random.sample(self.memory, min(trainSetSize, len(self.memory)))

        trainState = []
        trainAction = []
        trainNextState = []
        trainReward = []
        for i, (state, action, reward, next_state) in enumerate(trainSet):
            trainState.append(policy['normalizeStateForModelInput'](state))
            trainAction.append(action)
            trainNextState.append(policy['normalizeStateForModelInput'](next_state))
            trainReward.append(reward)
        # TODO: check trainAction contains action vectors rather than action ID

        trainState = np.array(trainState)
        trainAction = np.array(trainAction)
        trainReward = np.expand_dims(np.array(trainReward), axis=1)
        trainNextState = np.array(trainNextState)

        # Calculate targets
        # predict actor target model next state preferred action
        actorTargetPrediction = actor['targetModel'].predict(trainNextState)
        # predict critic target model (nextState, nextAction) Q value, i.e. future reward
        criticTargetPrediction = critic['targetModel'].predict([trainNextState, actorTargetPrediction])
        # update future **discounted** reward
        target = trainReward + (self.gamma * criticTargetPrediction)

        # train (update) critic train model
        loss = critic['trainModel'].train_on_batch([trainState, trainAction], target)
        # train (update) actor train model
        actions_for_grad = actor['trainModel'].predict(trainState)
        grads = critic['gradientsFunc'](trainState, actions_for_grad)
        actor['trainFunc'](trainState, grads)

        # update actor & critic target models weights
        updateTargetModelsParams()

        # update epsilon value
        actor['updateEpsilonFunc']()

        return loss
