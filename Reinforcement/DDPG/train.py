# -*- coding: utf-8 -*-
from Reinforcement.Functions import *
import json
from Actor import Actor
from Critic import Critic
from DeepNetwork import DeepNetwork

args = parseArguments()
dirName = createResultsFolder()
logger = initLogger(dirName)
# initGPU(args.gpuNum, args.gpuFrac)

# init info json file
info, jsonFullFname = loadInfoFile(dirName, logger)
info['args'] = vars(args)

# initialize policy and the agent
# policy =
# info['policy'] = policy.toJSON()

settings = None
with open(args.settings, 'r') as f:
    settings = json.load(f)

minGameScore = int(settings['minGameScoreRatio'] * settings['gameMinutesLength'])
settings['minGameScore'] = minGameScore
info['settings'] = settings

# init Actor
actor = Actor(1, 11, 8, settings['nModelBackups'])
# init Critic
critic = Critic(1, 11, 8, settings['nModelBackups'])

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
