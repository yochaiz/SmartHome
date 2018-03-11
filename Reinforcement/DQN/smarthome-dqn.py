# -*- coding: utf-8 -*-
from datetime import timedelta
from Reinforcement.Functions import *
import json
from DQNAgent import DQNAgent
from Reinforcement.DQN.WeekPolicyCNN import WeekPolicyCNN

args = parseArguments()
dirName = createResultsFolder()
logger = initLogger(dirName)
initGPU(args.gpuNum, args.gpuFrac)

# init info json file
info, jsonFullFname = loadInfoFile(dirName, logger)
info['args'] = vars(args)

settings = None
with open(args.settings, 'r') as f:
    settings = json.load(f)

minGameScore = int(settings['minGameScoreRatio'] * settings['gameMinutesLength'])
settings['minGameScore'] = minGameScore
info['settings'] = settings

# initialize policy and the agent
# policy = WeekPolicyLSTM("/home/yochaiz/SmartHome/Reinforcement/Policies/Week/policy2.json",settings['TAU'], 10)
policy = WeekPolicyCNN("/home/yochaiz/SmartHome/Reinforcement/Policies/Week/policy2.json", settings['TAU'], 10)
# policy = WeekPolicyFC("/home/yochaiz/SmartHome/Reinforcement/Policies/Week/policy2.json",settings['TAU'])
info['policy'] = policy.toJSON()

agent = DQNAgent(policy, settings['nModelBackups'], settings['dequeSize'])
info['agent'] = agent.toJSON()

# log info data
logInfo(info, logger)

# save info data to JSON
saveDataToJSON(info, jsonFullFname)

# save init model
agent.save(dirName, logger)

# print model to log
policy.getMainModel().summary(print_fn=lambda x: logger.info(x))

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
# init epsilon value
epsilon = agent.epsilon

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
    loss = 0
    for time_t in range(settings['gameMinutesLength']):
        # select action
        action, isRandom = agent.act(state)
        numOfRandomActions += isRandom

        # Advance the game to the next frame based on the action.
        next_state, reward = policy.step(state, action)
        score += reward

        # Remember the previous state, action, reward
        agent.remember(state, policy.actionToIdx(action), reward, next_state)

        # train network after each frame
        loss += agent.replay(settings['trainSetSize'], settings['batchSize'], settings['nEpochs'])

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

    logger.info("episode: {}, score:[{:.2f}], loss:[{:.5f}], sequence:[{}], random actions:[{}], eInit:[{:.4f}], init state:{}, end state:{}"
                .format(g, score, loss, curSequence, numOfRandomActions, epsilon, initState, state[-1, :]))

    # decrease game initial epsilon value
    epsilon = max(agent.epsilon_min, epsilon * agent.epsilon_decay)
    # update game initial epsilon value
    agent.epsilon = epsilon

    # stop playing if we reached the desired minimal game sequence
    if curSequence >= settings['minGameSequence']:
        break

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
