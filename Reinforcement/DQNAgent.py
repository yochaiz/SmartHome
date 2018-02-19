from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np


class DQNAgent:
    def __init__(self, logger, state_size, action_size, dequeLen=1000):
        self.logger = logger

        # self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=dequeLen)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # self.model = self._build_model()

        self.dictTypes = [str, int, float]

    # def _build_model(self):
    #     # Neural Net for Deep-Q learning Model
    #     model = Sequential()
    #     model.add(Dense(128, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(32, activation='relu'))
    #     model.add(Dense(16, activation='relu'))
    #     model.add(Dense(8, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     model.compile(loss='mse', optimizer='adam')
    #     # model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
    #     # model.compile(loss='mse', optimizer='sgd')
    #     model.summary(print_fn=lambda x: self.logger.info(x))
    #     model.summary()
    #     return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size), 1

        # Predict the reward value based on the given state
        input = state.astype(float)
        input[0][0] /= 24.0
        input[0][1] /= 60.0
        act_values = self.model.predict(input)
        # Pick the action based on the predicted reward
        return np.argmax(act_values[0]), 0

    def replay(self, trainSetSize, batchSize):
        # Sample trainSet from the memory
        trainSet = random.sample(self.memory, min(trainSetSize, len(self.memory)))

        # loss = 0

        def normalizeFeatures(input):
            input = input.astype(float)
            input[0] /= 24.0
            input[1] /= 60.0
            return input

        trainState = []
        trainAction = []
        trainNextState = []
        trainReward = []
        for state, action, reward, next_state in trainSet:
            trainState.append(normalizeFeatures(state[0]))
            trainAction.append(action)
            trainNextState.append(normalizeFeatures(next_state[0]))
            trainReward.append(reward)

        trainState = np.array(trainState)
        trainNextState = np.array(trainNextState)
        trainReward = np.array(trainReward)

        target = trainReward + (self.gamma * np.amax(self.model.predict(trainNextState), axis=1))
        target_f = self.model.predict(trainState)
        for i in range(len(target_f)):
            target_f[i, trainAction[i]] = target[i]

        scores = self.model.fit(trainState, target_f, batch_size=batchSize, epochs=1)
        loss = scores.history['loss'][0]

        # # Extract informations from each memory
        # for state, action, reward, next_state in trainSet:
        #     # print('state:{}, action:[{}], reward:[{}], next_state:{}'.format(state, action, reward, next_state))
        #     # target = reward
        #     # if not done:
        #     #     target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
        #
        #     # predict the future discounted reward
        #     # jj = self.model.predict(next_state)
        #     # print(jj)
        #     # jj = jj[0]
        #     # print(jj)
        #     input = next_state.astype(float)
        #     input[0][0] /= 24.0
        #     input[0][1] /= 60.0
        #     target = (reward + self.gamma * np.amax(self.model.predict(input)))
        #     # print('target:[{}]'.format(target))
        #     # make the agent to approximately map the current state to future discounted reward. We'll call that target_f
        #     input = state.astype(float)
        #     input[0][0] /= 24.0
        #     input[0][1] /= 60.0
        #     target_f = self.model.predict(input)
        #     # print('target_f:{}'.format(target_f))
        #     target_f[0][action] = target
        #     # print('target_f:{}'.format(target_f))
        #
        #     # Train the Neural Net with the state and target_f
        #     scores = self.model.fit(input, target_f, epochs=1, verbose=0)
        #     loss += scores.history['loss'][0]
        #
        # loss /= trainSetSize

        # update epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, dirName):
        fullPath = '{}/model.h5'.format(dirName)
        self.logger.info('Saving model as [{}]'.format(fullPath))
        # self.model.save_weights(name)
        self.model.save(fullPath)

    # convert class object to JSON serializable
    def toJSON(self):
        var = dict(vars(self))  # make dict copy
        keysToDelete = []
        for key, val in var.iteritems():
            if type(val) not in self.dictTypes:
                keysToDelete.append(key)

        for key in keysToDelete:
            del var[key]

        return var
