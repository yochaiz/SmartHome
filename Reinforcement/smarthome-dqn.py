# -*- coding: utf-8 -*-
import os
from datetime import timedelta, datetime
import logging
import argparse
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from Functions import loadInfoFile
import json
from DQNAgent import DQNAgent
from WeekPolicyLSTM import WeekPolicyLSTM
from WeekPolicy import WeekPolicy


# parse arguments
def parseArguments():
    parser = argparse.ArgumentParser(description='test model on dataset')
    parser.add_argument("gpuNum", type=int, help="GPU # to run on")
    parser.add_argument("--gpuFrac", type=float, default=0.3, help="GPU memory fraction")
    parser.add_argument("--settings", type=str, default='settings.json', help="Settings JSON file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequential", action='store_true', help="Init sequential state for a new game")
    group.add_argument("--random", action='store_true', help="Init random state for a new game")

    return parser.parse_args()


# init results directory
def createResultsFolder():
    rootDir = 'results'
    now = datetime.now()
    dirName = '{}/D-{}-{}-H-{}-{}-{}'.format(rootDir, now.day, now.month, now.hour, now.minute, now.second)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    return dirName


# initialize logger
def initLogger(dirName):
    logging.basicConfig(level=logging.INFO, filename=dirName + '/info.log')
    logger = logging.getLogger(__name__)

    return logger


# init GPU
def initGPU(gpuNum, gpuFrac):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuNum)

    # limit memory precentage usage
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpuFrac
    set_session(tf.Session(config=config))
    return


args = parseArguments()
dirName = createResultsFolder()
logger = initLogger(dirName)
initGPU(args.gpuNum, args.gpuFrac)

# init info json file
info, jsonFullFname = loadInfoFile(dirName, logger)
info['args'] = vars(args)

# initialize policy and the agent
# policy = WeekPolicyLSTM("Week_policies/policy2.json", 10)
policy = WeekPolicy("Week_policies/policy2.json")
info['policy'] = policy.toJSON()

settings = None
with open(args.settings, 'r') as f:
    settings = json.load(f)

minGameScore = int(settings['minGameScoreRatio'] * settings['gameMinutesLength'])
info['settings'] = settings

agent = DQNAgent(policy, settings['nModelBackups'], settings['dequeSize'])
info['agent'] = agent.toJSON()

# log info data
for key in info:
    logger.info('{}:[{}]'.format(key, info[key]))

# save info data to JSON
with open(jsonFullFname, 'w') as f:
    json.dump(info, f)

# save init model
agent.save(dirName, logger)

# print model to log
policy.model.summary(print_fn=lambda x: logger.info(x))

# initialize number of games
curSequence = 0
maxSequence = (0, [])
maxScore = (-1 * settings['gameMinutesLength'], [])
# Iterate the game
g = 0
# init start time
curTime = policy.timePrefixToDate(policy.generateRandomTimePrefix())
# init time delta
stateTimeDelta = timedelta(minutes=17)
while curSequence < settings['minGameSequence']:
    g += 1
    if args.random is True:
        state = policy.generateRandomInput()
    elif args.sequential is True:
        state = policy.buildDateInput(curTime)
        curTime += stateTimeDelta
    else:
        raise ValueError('Undefined init game state')

    initState = '{}'.format(state[-1, :])

    # time_t represents each minute of the game
    score = 0
    numOfRandomActions = 0
    for time_t in range(settings['gameMinutesLength']):
        # Decide action
        action, isRandom = agent.act(state)
        numOfRandomActions += isRandom

        # Advance the game to the next frame based on the action.
        next_state, reward = policy.step(state, action)
        score += reward

        # Remember the previous state, action, reward
        agent.remember(state, policy.actionToIdx(action), reward, next_state)

        # make next_state the new current state for the next frame.
        state = next_state

    # update current sequence length
    if score >= minGameScore:
        curSequence += 1
    else:
        curSequence = 0

    # update maximal score achieved during games
    if score > maxScore[0]:
        maxScore = (score, [g])
    elif abs(score - maxScore[0]) < 1E-5:
        maxScore[1].append(g)

    # update maximal sequence achieved during games
    if curSequence > maxSequence[0]:
        maxSequence = (curSequence, [g])
    elif abs(maxSequence[0] - curSequence) < 1E-5:
        maxSequence[1].append(g)

    # train network after game
    if curSequence < settings['minGameSequence']:
        loss = agent.replay(settings['trainSetSize'], settings['batchSize'], settings['nEpochs'])
    else:
        loss = 0

    logger.info("episode: {}, score:[{:.2f}], loss:[{:.5f}], sequence:[{}], random actions:[{}], e:[{:.4f}], init state:[{}]"
                .format(g, score, loss, curSequence, numOfRandomActions, agent.epsilon, initState))

    # save model and log max score & sequence values
    if (g % settings['nGamesPerSave']) == 0:
        agent.save(dirName, logger)
        logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))

# log max score & sequence values
logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))
		
# save model
agent.save(dirName, logger)

# class Settings:
#     def __init__(self, minGameScoreRatio, minGameSequence, gameMinutesLength, trainSetSize, batch_size):
#         self.minGameScore = int(minGameScoreRatio * gameMinutesLength)
#         self.minGameSequence = minGameSequence
#         self.gameMinutesLength = gameMinutesLength
#         self.trainSetSize = trainSetSize
#         self.batch_size = batch_size
