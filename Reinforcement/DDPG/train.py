# -*- coding: utf-8 -*-
from datetime import timedelta
from Reinforcement.Functions import *
from Reinforcement.Results import Results
from Actor import Actor
from Critic import Critic
from DeepNetwork import DeepNetwork
from ReplayBuffer import ReplayBuffer
from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy

args = parseArguments()
dirName = createResultsFolder()
logger = initLogger(dirName)
sess = initGPU(args.gpuNum, args.gpuFrac)

# init info json file
info, jsonFullFname = loadInfoFile(dirName, logger)
info['args'] = vars(args)

# initialize policy and the agent
policy = WeekPolicy("/home/yochaiz/SmartHome/Reinforcement/Policies/Week/policy1.json")
info['policy'] = policy.toJSON()

settings = None
with open(args.settings, 'r') as f:
    settings = json.load(f)

minGameScore = int(settings['minGameScoreRatio'] * settings['gameMinutesLength'])
settings['minGameScore'] = minGameScore
info['settings'] = settings

# init Results object
results = Results()
info['results'] = results.toJSON()

# init replay Buffer
replayBuffer = ReplayBuffer(settings['dequeSize'], settings['gamma'])
# init Actor
actor = Actor(sess, policy.idxToAction, policy.generateRandomAction, policy.normalizeStateForModelInput,
              policy.getStateDim(), policy.getActionDim(), settings['TAU'], settings['learningRate'], settings['nModelBackups'])
# init Critic
critic = Critic(sess, policy.getStateDim(), policy.getActionDim(), settings['TAU'], settings['learningRate'], settings['nModelBackups'])

# Log objects info to JSON
Loginfo = DeepNetwork.toJSON()
for key, value in Loginfo.iteritems():
    info[key] = value

# log info data
logInfo(info, logger)

# save info data to JSON
saveDataToJSON(info, jsonFullFname)

# save init models
DeepNetwork.save(dirName, logger)

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

while curSequence < settings['minGameSequence']:
    g += 1

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
    loss = 0
    for time_t in range(settings['gameMinutesLength']):
        # select action
        action, isRandom = actor.act(state, critic.getMainModel())
        numOfRandomActions += isRandom

        # Advance the game to the next frame based on the action.
        next_state, reward = policy.step(state, action)
        score += reward

        # Remember the previous state, action, reward, new state
        replayBuffer.remember(state, action, reward, next_state)

        # train network after each frame
        loss += replayBuffer.replay(actor.getMainModel(), actor.getTargetModel(), actor.train, actor.updateEpsilon, critic.getMainModel(),
                                    critic.getTargetModel(), critic.gradients, policy.normalizeStateForModelInput, DeepNetwork.updateModelParams,
                                    settings['batchSize'])
        # TODO: average loss ??

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

    # log game
    logger.info("episode: {}, score:[{:.2f}], loss:[{:.5f}], sequence:[{}], random actions:[{}], eInit:[{:.4f}], init state:{}, end state:{}"
                .format(g, score, loss, curSequence, numOfRandomActions, epsilon, initState, state[-1, :]))

    # update results object
    results.loss.append(round(loss, 5))
    results.score.append(int(score))

    # update info data in JSON, add results
    info['results'] = results.toJSON()
    saveDataToJSON(info, jsonFullFname)

    # decrease game initial epsilon value
    epsilon = max(actor.epsilon_min, epsilon * actor.epsilon_decay)
    # update game initial epsilon value
    actor.epsilon = epsilon

    # stop playing if we reached the desired minimal game sequence
    if curSequence >= settings['minGameSequence']:
        break

    # save models and log max score & sequence values
    if (g % settings['nGamesPerSave']) == 0:
        DeepNetwork.save(dirName, logger)
        logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))

## GAME HAS ENDED
# log max score & sequence values
logger.info("maxScore:{} , maxSequence:{}".format(maxScore, maxSequence))

# save models
DeepNetwork.save(dirName, logger)
