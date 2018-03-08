# -*- coding: utf-8 -*-
from datetime import timedelta
from Reinforcement.Functions import *
import json
from DQNAgent import DQNAgent
from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy

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
settings['minGameScore'] = minGameScore
info['settings'] = settings

agent = DQNAgent(policy, settings['nModelBackups'], settings['dequeSize'])
info['agent'] = agent.toJSON()

# log info data
logInfo(info, logger)

# save info data to JSON
saveDataToJSON(info, jsonFullFname)

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
        # select action
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
    maxScore = updateMaxTuple(score, g, maxScore)
    # update maximal sequence achieved during games
    maxSequence = updateMaxTuple(curSequence, g, maxSequence)

    # train network after game
    if curSequence < settings['minGameSequence']:
        loss = agent.replay(settings['trainSetSize'], settings['batchSize'], settings['nEpochs'])
    else:
        loss = 0

    logger.info("episode: {}, score:[{:.2f}], loss:[{:.5f}], sequence:[{}], random actions:[{}], e:[{:.4f}], init state:{}, end state:{}"
                .format(g, score, loss, curSequence, numOfRandomActions, agent.epsilon, initState, state[-1, :]))

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
