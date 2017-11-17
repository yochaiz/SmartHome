# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import os
from datetime import time, timedelta, datetime, date
from random import randint
import logging
import argparse
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from Functions import loadInfoFile
import json


class DQNAgent:
    def __init__(self, state_size, action_size, dequeLen=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.dequeLen = dequeLen
        self.memory = deque(maxlen=dequeLen)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mse', optimizer='adam')
        # model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        # model.compile(loss='mse', optimizer='sgd')
        model.summary(print_fn=lambda x: logger.info(x))
        model.summary()
        return model

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

    def replay(self, batch_size):
        # Sample minibatch from the memory
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        loss = 0

        # Extract informations from each memory
        for state, action, reward, next_state in minibatch:
            # print('state:{}, action:[{}], reward:[{}], next_state:{}'.format(state, action, reward, next_state))
            # target = reward
            # if not done:
            #     target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            # predict the future discounted reward
            # jj = self.model.predict(next_state)
            # print(jj)
            # jj = jj[0]
            # print(jj)
            input = next_state.astype(float)
            input[0][0] /= 24.0
            input[0][1] /= 60.0
            target = (reward + self.gamma * np.amax(self.model.predict(input)))
            # print('target:[{}]'.format(target))
            # make the agent to approximately map the current state to future discounted reward. We'll call that target_f
            input = state.astype(float)
            input[0][0] /= 24.0
            input[0][1] /= 60.0
            target_f = self.model.predict(input)
            # print('target_f:{}'.format(target_f))
            target_f[0][action] = target
            # print('target_f:{}'.format(target_f))

            # Train the Neural Net with the state and target_f
            scores = self.model.fit(input, target_f, epochs=1, verbose=0)
            loss += scores.history['loss'][0]

        loss /= batch_size

        # update epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # self.model.save_weights(name)
        self.model.save(name)


class Policy:
    # a policy for a single light over a single day.
    # the policy is that the light is ON only between 6:00-18:00.
    # the rest of the day, the light is OFF.
    # state structure: [Hour,Minute,Light state {1=on,0=off}]
    # possible actions: ['Don't change status' , 'Change light status' (i.e. turn it on if off)]
    # startTime = time(hour=6, minute=0)
    # endTime = time(hour=18, minute=0)

    def __init__(self, startTime, endTime):
        self.stateSize = 3
        self.actionSize = 2
        self.startTime = startTime
        self.endTime = endTime

    @staticmethod
    def calcMinutesGap(time1, time2):
        if time1 > time2:
            endTime = time1
            startTime = time2
        else:
            endTime = time2
            startTime = time1

        duration = datetime.combine(date.min, endTime) - datetime.combine(date.min, startTime)
        return duration.seconds / 60

    # perform action
    def step(self, state, action):
        curDate = datetime(year=2000, month=1, day=1, hour=state[0], minute=state[1])
        nextDate = curDate + timedelta(minutes=1)
        # update light state at newState due to action performed
        nextState = np.array([nextDate.hour, nextDate.minute, (state[2] + action) % self.actionSize])

        # print(state)
        # print(action)
        # print(nextState)
        # print(curDate.time())
        # print(nextDate.time())
        # print(self.startTime)
        # print(self.endTime)

        positiveReward = 1
        negativeReward = -1

        # light is on
        if state[2] == 1:
            # light should be on
            if self.startTime <= curDate.time() <= self.endTime:
                if action == 0:  # agent doesn't change status
                    reward = positiveReward
                else:  # agent changes status (turns off)
                    reward = negativeReward  # * (self.calcMinutesGap(self.endTime, curDate.time()))

            # light should be off
            else:
                if action == 0:  # agent doesn't change status
                    reward = negativeReward  # * (self.calcMinutesGap(self.startTime, curDate.time()))
                else:  # agent changes status (turns off)
                    reward = positiveReward

        # light is off
        else:
            # light should be on
            if self.startTime <= curDate.time() <= self.endTime:
                if action == 0:  # agent doesn't change status
                    reward = negativeReward  # * (self.calcMinutesGap(self.endTime, curDate.time()))
                else:  # agent changes status (turns off)
                    reward = positiveReward

            # light should be off
            else:
                if action == 0:  # agent doesn't change status
                    reward = positiveReward
                else:  # agent changes status (turns off)
                    reward = negativeReward  # * (self.calcMinutesGap(self.startTime, curDate.time()))

        # print('Time:[{}:{}] , Light state:[{}] , Action:[{}] , Reward:[{}]'.format(curDate.hour, curDate.minute, state[2], action, reward))
        return nextState, reward

    def getRandomState(self):
        hour = randint(0, 23)
        minute = randint(0, 59)
        randTime = time(hour=hour, minute=minute)
        state = np.array([hour, minute, 0])
        if self.startTime <= randTime <= self.endTime:
            state[2] = 1  # light is on

        return state


class Settings:
    def __init__(self, minGameScoreRatio, minGameSequence, gameMinutesLength, batch_size):
        self.minGameScore = int(minGameScoreRatio * gameMinutesLength)
        self.minGameSequence = minGameSequence
        self.gameMinutesLength = gameMinutesLength
        self.batch_size = batch_size


# parse arguments
parser = argparse.ArgumentParser(description='test model on dataset')
parser.add_argument("gpuNum", type=int, help="GPU # to run on")
parser.add_argument("--gpuFrac", type=float, default=0.3, help="GPU memory fraction")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--sequential", action='store_true', help="Init sequential state for a new game")
group.add_argument("--random", action='store_true', help="Init random state for a new game")
args = parser.parse_args()

# init result directory
rootDir = 'results'
now = datetime.now()
dirName = '{}/D-{}-{}-H-{}-{}'.format(rootDir, now.day, now.month, now.hour, now.minute)
if not os.path.exists(dirName):
    os.makedirs(dirName)

# initialize logger
logging.basicConfig(level=logging.INFO, filename=dirName + '/info.log')
logger = logging.getLogger(__name__)

# init GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuNum)

# limit memory precentage usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpuFrac
set_session(tf.Session(config=config))

# init info json file
info, jsonFullFname = loadInfoFile(dirName, logger)
info['args'] = vars(args)

# initialize policy and the agent
policy = Policy(time(hour=6, minute=0), time(hour=18, minute=0))
info['policy'] = vars(policy)
info['policy']['startTime'] = str(info['policy']['startTime'])
info['policy']['endTime'] = str(info['policy']['endTime'])
state_size = policy.stateSize
action_size = policy.actionSize

agent = DQNAgent(state_size, action_size)
info['agent'] = vars(agent)
del info['agent']['model']
del info['agent']['memory']

# each game length is 5 hours (300 minutes)
settings = Settings(0.85, 50, 500, 128)
info['settings'] = vars(settings)

logger.info('info:[{}]'.format(info))
with open(jsonFullFname, 'w') as f:
    json.dump(info, f)

# done = False

# initialize number of games
curSequence = 0
# Iterate the game
g = 0
stateTime = time(hour=8, minute=0)
stateTimeDelta = timedelta(minutes=17)
while curSequence < settings.minGameSequence:
    g += 1
    # reset state in the beginning of each game
    # state = np.array([13, 30, 0])
    if args.random is True:
        state = policy.getRandomState()
    elif args.sequential is True:
        state = np.array([stateTime.hour, stateTime.minute, random.randrange(2)])
        stateTime = datetime(year=2000, month=1, day=1, hour=stateTime.hour, minute=stateTime.minute)
        stateTime += stateTimeDelta
        stateTime = stateTime.time()
    else:
        raise ValueError('Undefined init game state')

    initState = 'init state:[{}]'.format(state)
    state = np.reshape(state, [1, state_size])

    # time_t represents each minute of the game
    score = 0
    numOfRandomActions = 0
    for time_t in range(settings.gameMinutesLength):
        # Decide action
        action, isRandom = agent.act(state)
        numOfRandomActions += isRandom

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward = policy.step(state[0], action)
        next_state = np.reshape(next_state, [1, state_size])
        score += reward

        # Remember the previous state, action, reward
        agent.remember(state, action, reward, next_state)

        # make next_state the new current state for the next frame.
        state = next_state

    if score >= settings.minGameScore:
        curSequence += 1
    else:
        curSequence = 0

    loss = 'No training'
    if curSequence < settings.minGameSequence:
        loss = agent.replay(settings.batch_size)

    logger.info(
        "episode: {}, state: {}, score: [{}], loss:[{}], sequence:[{}], random actions:[{}], e:[{:.2}]".format(g, initState, score, loss, curSequence, numOfRandomActions,
                                                                                                               agent.epsilon))

modelName = 'model.h5'
logger.info('Saving model as [{}]'.format(rootDir + dirName + '/' + modelName))
agent.save(rootDir + dirName + '/' + modelName)
