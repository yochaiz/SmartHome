# -*- coding: utf-8 -*-
from Functions import *
import json

args = parseArguments()
dirName = createResultsFolder()
logger = initLogger(dirName)
initGPU(args.gpuNum, args.gpuFrac)

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

# actor =
# info['actor'] = actor.toJSON()

# critic =
# info['critic'] = critic.toJSON()

# log info data
logInfo(info, logger)

# save info data to JSON
saveDataToJSON(info, jsonFullFname)

# save init models
# actor.save(dirName, logger)
# critic.save(dirName, logger)

# print models to log
# actor.printModel(logger)
# critic.printModel(logger)
