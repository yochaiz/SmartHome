from collections import deque
import random
import numpy as np


class DQNAgent:
    dictTypes = [str, int, float]

    def __init__(self, policy, nBackups=3, dequeLen=1000):
        self.policy = policy
        self.nBackups = nBackups
        self.curBackupIdx = 0

        self.memory = deque(maxlen=dequeLen)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1 - 1E-4
        self.learning_rate = 0.001

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return self.policy.generateRandomAction(), 1

        # Predict the reward value based on the given state
        input = self.policy.normalizeStateForModelInput(state)
        input = np.expand_dims(input, axis=0)
        actValues = self.policy.model.predict(input)
        actionIdx = np.argmax(actValues)
        action = self.policy.idxToAction(actionIdx)
        # Pick the action based on the predicted reward
        return action, 0

    def replay(self, trainSetSize, batchSize):
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

        trainState = np.array(trainState)
        trainReward = np.array(trainReward)
        trainNextState = np.array(trainNextState)

        # predict the future discounted reward
        futureReward = self.policy.model.predict(trainNextState)
        futureReward = np.amax(futureReward, axis=2)
        futureReward = futureReward[:, 0]

        target = trainReward + (self.gamma * futureReward)

        # make the agent to approximately map the current state to future discounted reward
        target_f = self.policy.model.predict(trainState)
        for i in range(len(target_f)):
            target_f[i, -1, trainAction[i]] = target[i]

        scores = self.policy.model.fit(trainState, target_f, batch_size=batchSize, epochs=1, verbose=0)
        loss = scores.history['loss'][0]

        # # Extract informations from each memory
        # for state, action, reward, next_state in trainSet:
        #     # print('state:{}, action:[{}], reward:[{}], next_state:{}'.format(state, action, reward, next_state))
        #     # target = reward
        #     # if not done:
        #     #     target = (reward + self.gamma * np.amax(self.policy.model.predict(next_state)[0]))
        #
        #     # predict the future discounted reward
        #     # jj = self.policy.model.predict(next_state)
        #     # print(jj)
        #     # jj = jj[0]
        #     # print(jj)
        #     input = next_state.astype(float)
        #     input[0][0] /= 24.0
        #     input[0][1] /= 60.0
        #     target = (reward + self.gamma * np.amax(self.policy.model.predict(input)))
        #     # print('target:[{}]'.format(target))
        #     # make the agent to approximately map the current state to future discounted reward. We'll call that target_f
        #     input = state.astype(float)
        #     input[0][0] /= 24.0
        #     input[0][1] /= 60.0
        #     target_f = self.policy.model.predict(input)
        #     # print('target_f:{}'.format(target_f))
        #     target_f[0][action] = target
        #     # print('target_f:{}'.format(target_f))
        #
        #     # Train the Neural Net with the state and target_f
        #     scores = self.policy.model.fit(input, target_f, epochs=1, verbose=0)
        #     loss += scores.history['loss'][0]
        #
        # loss /= trainSetSize

        # update epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def load(self, name):
        self.policy.model.load_weights(name)

    def save(self, dirName, logger):
        fullPath = '{}/model-{}.h5'.format(dirName, self.curBackupIdx)
        logger.info('Saving model as [{}]'.format(fullPath))
        self.policy.model.save(fullPath)
        # update next save index
        self.curBackupIdx = (self.curBackupIdx + 1) % self.nBackups

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
