from collections import deque
import numpy as np
import random
import heapq


class MaxHeapObj(object):
    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class MinHeap(object):
    def __init__(self):
        self.h = []

    def heappush(self, x):
        heapq.heappush(self.h, x)

    def heappop(self):
        return heapq.heappop(self.h)

    def __getitem__(self, i):
        return self.h[i]

    def __len__(self):
        return len(self.h)

    def remove(self, v):
        self.h.remove(v)

    def heaptop(self):
        if len(self.h) > 0:
            return self.h[0]
        else:
            return None


class MaxHeap(MinHeap):
    def heappush(self, x):
        heapq.heappush(self.h, MaxHeapObj(x))

    def heappop(self):
        return heapq.heappop(self.h).val

    def __getitem__(self, i):
        return self.h[i].val

    def remove(self, v):
        self.h.remove(MaxHeapObj(v))


class ReplayBuffer:
    def __init__(self, size, gamma):
        # init deque
        self.memory = deque(maxlen=size)
        # init heaps
        self.goodReward = MinHeap()
        self.badReward = MaxHeap()

        self.gamma = gamma

    def remember(self, state, action, reward, next_state):
        element = (reward, state, action, next_state)

        if len(self.memory) == self.memory.maxlen:
            vOut = self.memory[0]
            # remove oldest deque element from heaps
            if vOut >= self.goodReward.heaptop():
                self.goodReward.remove(vOut)
            else:
                self.badReward.remove(vOut)

        # add element to the relevant heap
        if (len(self.goodReward) > 0) and (element >= self.goodReward.heaptop()):
            self.goodReward.heappush(element)
        else:
            self.badReward.heappush(element)

        # balance heaps
        if abs(len(self.goodReward) - len(self.badReward)) > 1:
            if len(self.goodReward) > len(self.badReward):
                vOut = self.goodReward.heappop()
                self.badReward.heappush(vOut)
            else:
                vOut = self.badReward.heappop()
                self.goodReward.heappush(vOut)

        # add element to deque
        self.memory.append(element)

    def replay(self, actorMainModel, actorTargetModel, actorTrainFunc, actorWolpertingerFunc, actorUpdateEpsilonFunc,
               criticMainModel, criticTargetModel, criticGradientsFunc, stateNormalizeFunc, updateTargetModelsParams,
               trainSetSize):
        # Sample trainSet from  memory
        # trainSet = random.sample(self.memory, min(trainSetSize, len(self.memory)))
        trainSet = random.sample(self.goodReward.h, min(int(trainSetSize / 2), len(self.goodReward)))
        tempSet = random.sample(self.badReward.h, min(trainSetSize - len(trainSet), len(self.badReward)))
        trainSet.extend([t.val for t in tempSet])
        del tempSet

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
