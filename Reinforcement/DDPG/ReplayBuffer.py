from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, actor, critic, trainSetSize):
        # Sample trainSet from the memory
        trainSet = random.sample(self.memory, min(trainSetSize, len(self.memory)))

        trainState = []
        trainAction = []
        trainNextState = []
        trainReward = []
        for i, (state, action, reward, next_state) in enumerate(trainSet):
            trainState.append(self.policy.normalizeStateForModelInput(state))
            trainAction.append(action)
            trainNextState.append(self.policy.normalizeStateForModelInput(next_state))
            trainReward.append(reward)
        # TODO: check trainAction contains action vectors rather than action ID

        trainState = np.array(trainState)
        trainReward = np.array(trainReward)
        trainNextState = np.array(trainNextState)

        # Calculate targets
        # predict actor target model next state preferred action
        actorTargetPrediction = actor.models[actor.targetModelKey].predict(trainNextState)
        # predict critic target model (nextState, nextAction) Q value, i.e. future reward
        criticTargetPrediction = critic.models[critic.targetModelKey].predict([trainNextState, actorTargetPrediction])
        # update future **discounted** reward
        target = trainReward + (self.gamma * criticTargetPrediction)

        # train (update) critic train model
        loss = critic.models[critic.trainModelKey].train_on_batch([trainState, trainAction], target)
        # train (update) actor train model
        actions_for_grad = actor.models[actor.trainModelKey].predict(trainState)
        grads = critic.gradients(trainState, actions_for_grad)
        actor.train(trainState, grads)

        # update actor & critic target models weights



        # update epsilon value
        actor.epsilon = max(actor.epsilon_min, (actor.epsilon * actor.epsilon_decay))

        return loss
