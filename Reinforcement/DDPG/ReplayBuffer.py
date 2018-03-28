from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, size, gamma):
        self.memory = deque(maxlen=size)
        self.gamma = gamma

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, actorMainModel, actorTargetModel, actorTrainFunc, actorWolpertingerFunc, actorUpdateEpsilonFunc,
               criticMainModel,
               criticTargetModel, criticGradientsFunc, stateNormalizeFunc, updateTargetModelsParams, trainSetSize):
        # Sample trainSet from the memory
        trainSet = random.sample(self.memory, min(trainSetSize, len(self.memory)))

        trainState = []
        trainAction = []
        trainNextState = []
        trainReward = []
        for i, (state, action, reward, next_state) in enumerate(trainSet):
            trainState.append(stateNormalizeFunc(state))
            trainAction.append(action)
            trainNextState.append(stateNormalizeFunc(next_state))
            trainReward.append(reward)
        # TODO: check trainAction contains action vectors rather than action ID

        trainState = np.array(trainState)
        trainAction = np.array(trainAction)
        trainReward = np.expand_dims(np.array(trainReward), axis=1)
        trainNextState = np.array(trainNextState)

        # Calculate targets
        # # predict actor target model next state preferred continuous action
        # action = actorTargetModel.predict(trainNextState)
        # predict actor target model next state preferred discrete action
        action, _, _ = actorWolpertingerFunc(trainNextState, actorTargetModel, criticTargetModel)
        # TODO: it is not clear from paper if critic model should be MAIN or TARGET ???
        # predict critic target model (nextState, nextAction) Q value, i.e. future reward
        criticTargetPrediction = criticTargetModel.predict([trainNextState, action])
        # update future **discounted** reward
        target = trainReward + (self.gamma * criticTargetPrediction)

        # train (update) critic train model
        loss = criticMainModel.train_on_batch([trainState, trainAction], target)
        # train (update) actor train model
        actions_for_grad = actorMainModel.predict(trainState)
        grads = criticGradientsFunc(trainState, actions_for_grad)
        actorTrainFunc(trainState, grads)

        # update actor & critic target models weights
        updateTargetModelsParams()

        # update epsilon value
        actorUpdateEpsilonFunc()

        return loss
