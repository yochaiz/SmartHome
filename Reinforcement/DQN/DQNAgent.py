from collections import deque
import random
import numpy as np


class DQNAgent:
    dictTypes = [str, int, float]

    def __init__(self, policy, nBackups, dequeLen):
        self.policy = policy
        self.nBackups = nBackups
        self.curBackupIdx = 0

        self.memory = deque(maxlen=dequeLen)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # 1 - 1E-3
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
        actValues = self.policy.getMainModel().predict(input)
        actionIdx = np.argmax(actValues)
        action = self.policy.idxToAction(actionIdx)
        # Pick the action based on the predicted reward
        return action, 0

    def replay(self, trainSetSize, batchSize, nEpochs):
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
        futureReward = self.policy.getTargetModel().predict(trainNextState)
        futureReward = np.amax(futureReward, axis=1)

        target = trainReward + (self.gamma * futureReward)

        # make the agent to approximately map the current state to future discounted reward
        target_f = self.policy.getMainModel().predict(trainState)
        # diff = 0
        for i in range(len(target_f)):
            # diff += pow((target[i] - target_f[i, trainAction[i]]), 2)
            target_f[i, trainAction[i]] = target[i]

        # diff /= len(target_f)

        scores = self.policy.getMainModel().fit(trainState, target_f, batch_size=batchSize, epochs=nEpochs, verbose=0)
        loss = scores.history['loss'][0]

        # update actor & critic target models weights
        self.policy.updateModelParams()

        # update epsilon value
        self.epsilon = max(self.epsilon_min, (self.epsilon * self.epsilon_decay))

        return loss

    def save(self, dirName, logger):
        fullPath = '{}/model-{}.h5'.format(dirName, self.curBackupIdx)
        logger.info('Saving model as [{}]'.format(fullPath))
        self.policy.getMainModel().save(fullPath)
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

    # def load(self, name):
    #     self.policy.model.load_weights(name)

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
