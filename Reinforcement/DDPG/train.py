# -*- coding: utf-8 -*-
from datetime import timedelta
import time, json, os, inspect
import numpy as np
from shutil import copy2
import Reinforcement.Functions as Funcs
from Reinforcement.Results import Results
from Reinforcement.DDPG.Actor import Actor
from Reinforcement.DDPG.Critic import Critic
from Reinforcement.DDPG.DeepNetwork import DeepNetwork
from Reinforcement.DDPG.ReplayBuffer import ReplayBuffer
from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy

# init current file (script) folder
baseFolder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
# init policy
policy = WeekPolicy("/home/yochaiz/SmartHome/Reinforcement/Policies/Week/policy2.json")
# init results
results = Results(baseFolder, policy.getActionDim())

args = Funcs.parseArguments()
dirName = results.getFullPath()
logger = Funcs.initLogger(dirName)
sess = Funcs.initGPU(args.gpuNum, args.gpuFrac)
Funcs.attachSIGTERMhandler(results, logger)
# save source code
Funcs.saveCode(dirName, (baseFolder, ['train.py', 'Actor.py', 'Critic.py', 'DeepNetwork.py', 'ReplayBuffer.py']))

# init info json file
info, jsonFullFname = Funcs.loadInfoFile(dirName, logger)
info['args'] = vars(args)
info['policy'] = policy.toJSON()
info['results'] = results.toJSON()

# save policy JSON file to training results folder
copy2(policy.getFname(), '{}/policy.json'.format(dirName))

settings = None
with open(args.settings, 'r') as f:
    settings = json.load(f)

minGameScore = int(settings['minGameScoreRatio'] * settings['gameMinutesLength'] * policy.rewardScaleFactor)
settings['minGameScore'] = minGameScore
info['settings'] = settings

# init replay Buffer
replayBuffer = ReplayBuffer(settings['dequeSize'], settings['gamma'])
# init Actor
actor = Actor(sess, policy.idxToAction, policy.generateRandomAction, policy.normalizeStateForModelInput,
              policy.getStateDim(), policy.getActionDim(), settings['TAU'], settings['learningRate'], args.k,
              settings['nModelBackups'])
# set epsilon reset signal
actor.setEpsilonHandler(logger)

# init Critic
critic = Critic(sess, policy.getStateDim(), policy.getActionDim(), settings['TAU'], settings['learningRate'],
                settings['nModelBackups'])

# Log objects info to JSON
Loginfo = DeepNetwork.toJSON()
for key, value in Loginfo.items():
    info[key] = value

# log info data
Funcs.logInfo(info, logger)

# save info data to JSON
Funcs.saveDataToJSON(info, jsonFullFname)

# save init models
DeepNetwork.save(dirName, logger)

# print descriptions
desc = [("General", [args.desc])]
desc.extend(DeepNetwork.getDescLogs())
Funcs.logDescriptions(logger, desc)

# print models to log
DeepNetwork.printModel(logger)

# init game params
curSequence = 0
maxSequence = (0, [])
maxScore = (-1 * settings['gameMinutesLength'], [])
g = 0  # game number
curTime = policy.timePrefixToDate(policy.generateRandomTimePrefix())  # init start time
stateTimeDelta = timedelta(minutes=17)  # init time delta
epsilon = actor.epsilon
# init optimal models save history array for logging
optimalModelsHistory = []

while curSequence < settings['minGameSequence']:
    g += 1
    startT = time.time()

    # set game init state
    if args.random is True:
        state = policy.generateRandomInput()
    elif args.sequential is True:
        state = policy.buildDateInput(curTime)
        curTime += stateTimeDelta
    else:
        raise ValueError('Undefined init game state')

    # save game init state as string for game log
    initState = '{}'.format(state[-1, :])

    # time_t represents each minute of the game
    score = 0
    numOfRandomActions = 0
    isInPoolRatio = 0
    numOfOptActionSelected = 0
    optActionInPoolButNotSelected = 0
    loss = 0
    for time_t in range(settings['gameMinutesLength']):
        # build optimal action for comparison purposes
        curDate = policy.timePrefixToDate(state)
        nextDate = curDate + policy.minTimeUnit()
        optimalNextState = policy.buildExpectedState(nextDate)
        optimalAction = np.logical_xor(state[-1, -policy.numOfDevices:], optimalNextState).astype(int)

        # select action
        action, isRandom, isInPool, isOptActionSelected = actor.act(state, critic.getMainModel(), optimalAction)
        numOfRandomActions += isRandom
        isInPoolRatio += isInPool
        numOfOptActionSelected += isOptActionSelected
        optActionInPoolButNotSelected += abs(isInPool - isOptActionSelected) if (isRandom == 0) else 0

        # Advance the game to the next frame based on the action.
        next_state, reward = policy.step(state, action)
        score += reward

        # Remember the previous state, action, reward, new state
        replayBuffer.remember(state, action, reward, next_state)

        # train network after each frame
        loss += replayBuffer.replay(actor.getMainModel(), actor.getTargetModel(), actor.train, actor.wolpertingerAction,
                                    actor.updateEpsilon, critic.getMainModel(), critic.getTargetModel(),
                                    critic.gradients, policy.normalizeStateForModelInput, DeepNetwork.updateModelParams,
                                    settings['trainSetSize'])

        # make next_state the new current state for the next frame.
        state = next_state

    optActionInPoolButNotSelected /= float(isInPoolRatio)
    isInPoolRatio /= float(settings['gameMinutesLength'] - numOfRandomActions)
    numOfOptActionSelected /= float(settings['gameMinutesLength'])

    # update current sequence length
    if score >= minGameScore:
        curSequence += 1
    else:
        curSequence = 0

    endT = time.time()

    # log game
    logger.info(
        "episode: {}, score:[{:.2f}], loss:[{:.5f}], sequence:[{}], isInPoolRatio:[{:.2f}], optActionSelectedRatio:[{:.2f}], "
        "optActionInPoolButNotSelected:[{:.2f}], random actions:[{}], eInit:[{:.4f}], init state:{}, end state:{}, runtime(seconds):[{:.2f}]"
            .format(g, score, loss, curSequence, isInPoolRatio, numOfOptActionSelected, optActionInPoolButNotSelected,
                    numOfRandomActions,
                    epsilon, initState, state[-1, :], (endT - startT)))

    # update opt model if achieved new max score
    if score > maxScore[0]:
        backupIdx = DeepNetwork.save(dirName, None)
        optimalModelsHistory.append((g, backupIdx))
    # update maximal score achieved during games
    maxScore = Funcs.updateMaxTuple(score, g, maxScore)
    # update maximal sequence achieved during games
    maxSequence = Funcs.updateMaxTuple(curSequence, g, maxSequence)

    # update results object
    results.loss.append(round(loss, 5))
    results.score.append(int(score))

    # update info data in JSON, add results
    info['results'] = results.toJSON()
    Funcs.saveDataToJSON(info, jsonFullFname)

    # decrease game initial epsilon value
    epsilon = max(actor.epsilon_min, epsilon * actor.epsilon_decay)
    # update game initial epsilon value
    actor.epsilon = epsilon

    # stop playing if we reached the desired minimal game sequence
    if curSequence >= settings['minGameSequence']:
        break

    # save models and log max score & sequence values
    if (g % settings['nGamesPerSave']) == 0:
        # save models if haven't saved them lately
        if len(optimalModelsHistory) == 0:
            DeepNetwork.save(dirName, logger)
        # log max score & sequence values
        logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))
        # log when we have reached optimal models lately
        logger.info("Optimal models save history:{}".format(optimalModelsHistory))
        optimalModelsHistory = []

## GAME HAS ENDED
# log max score & sequence values
logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))
# save models
DeepNetwork.save(dirName, logger)
# move to ended folder
results.moveToEnded()
