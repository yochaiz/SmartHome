import argparse
import os
import logging
import matplotlib.pyplot as plt
import json
from Functions import loadInfoFile


def plot(info, folderName):
    if info is not None:
        plt.plot(info['scores'], 'bo')
        plt.plot([info['minGameScore']] * len(info['scores']), 'r--')
        plt.xlabel('Game no.')
        plt.ylabel('Score')
        plt.title('[{}]'.format(folderName))
        plt.legend(['Game score', 'min Game Score'], bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


def addToInfo(info, key, value):
    logger.info('Adding key [{}] to info'.format(key))
    info[key] = value
    return info


def loadPlotInfo(jsonFullFname, info, keys):
    infoModified = False

    fname = 'info.log'
    fnameFullPath = '{}/{}'.format(args.folderName, fname)
    if os.path.exists(fnameFullPath):
        with open(fnameFullPath, 'r') as f:
            for key, func in keys.iteritems():
                if key not in info:
                    info = addToInfo(info, key, func(f))
                    infoModified = True

            if infoModified is True:
                logger.info('Writing to [{}]'.format(jsonFullFname))
                with open(jsonFullFname, 'w') as fw:
                    json.dump(info, fw)

    return info


def extractMinGameScoreFromFile(file):
    v = None
    for line in file.readlines():
        idx = line.find('minGameScore')
        if idx >= 0:
            idx = line.find('[', idx)
            if idx >= 0:
                idx2 = line.find(']', idx)
                v = eval(line[(idx + 1):idx2])
                break

    return v


def extractScoresFromFile(file):
    scores = []
    file.seek(0)

    for line in file.readlines():
        idx = line.find('score')
        if idx >= 0:
            idx = line.find('[', idx)
            if idx >= 0:
                idx2 = line.find(']', idx)
                v = line[(idx + 1):idx2]
                scores.append(eval(v))

    return scores

    # parse arguments


parser = argparse.ArgumentParser(description='test model on dataset')
parser.add_argument("folderName", type=str, help="Folder name where the model for testing is located")
args = parser.parse_args()

# initialize logger
logging.basicConfig(level=logging.INFO, filename=args.folderName + '/plot.log')
logger = logging.getLogger(__name__)
logger.info('args:[{}]'.format(args))

info, jsonFullFname = loadInfoFile(args.folderName,logger)
keys = {'scores': extractScoresFromFile, 'minGameScore': extractMinGameScoreFromFile}
info = loadPlotInfo(jsonFullFname, info, keys)
plot(info, args.folderName)
