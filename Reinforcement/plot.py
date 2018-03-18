import argparse
import logging
import matplotlib.pyplot as plt
from pylab import savefig
from Functions import loadInfoFile, CollectResultsFromLog


def updateMinVal(val, scale):
    val -= abs(scale * val)
    return val


def updateMaxVal(val, scale):
    val += abs(scale * val)
    return val


# data - a list of tuples (points, style&color, label)
def plot(data, yLabel, folderName):
    fig = plt.figure()
    labels = []
    yRange = [min(data[0][0]), max(data[0][0])]

    for (points, color, label) in data:
        plt.plot(points, color)
        labels.append(label)
        yRange[0] = min(yRange[0], min(points))
        yRange[1] = max(yRange[1], max(points))

    # add some margins to Y axis
    scale = 0.1
    yRange[0] = updateMinVal(yRange[0], scale)
    yRange[1] = updateMaxVal(yRange[1], scale)
    for i in range(len(yRange)):
        yRange[i] = int(round(yRange[i]))

    plt.xlabel('Game no.')
    plt.ylabel(yLabel)
    plt.ylim(yRange[0], yRange[1])
    plt.title('[{}]'.format(folderName))
    plt.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()

    savefig('{}/{}.png'.format(folderName, yLabel))


# parse arguments
parser = argparse.ArgumentParser(description='test model on dataset')
parser.add_argument("folderName", type=str, help="Folder name where the model for testing is located")
args = parser.parse_args()

# initialize logger
logging.basicConfig(level=logging.INFO, filename=args.folderName + '/plot.log')
logger = logging.getLogger(__name__)
logger.info('args:[{}]'.format(args))

resultsKeyName = 'results'
# load info JSON
info, jsonFullFname = loadInfoFile(args.folderName, logger)
# if there are no results in JSON, add them to JSON
if resultsKeyName not in info.keys():
    info = CollectResultsFromLog(args.folderName)

# plot loss
key = 'loss'
points = info[resultsKeyName][key]
plot([(points, 'bo', key)], key, args.folderName)

# plot scores
key = 'score'
points = info[resultsKeyName][key]
plot([(points, 'bo', key), ([info['settings']['minGameScore']] * len(points), 'r--', 'minGameScore'),
      ([info['settings']['gameMinutesLength']] * len(points), 'g--', 'maxPossibleScore')], key, args.folderName)

# keys = {'scores': extractScoresFromFile, 'minGameScore': extractMinGameScoreFromFile}
# info = loadPlotInfo(jsonFullFname, info, keys)
# plot(info, args.folderName)


# def plot(info, folderName):
#     if info is not None:
#         plt.plot(info['scores'], 'bo')
#         plt.plot([info['minGameScore']] * len(info['scores']), 'r--')
#         plt.xlabel('Game no.')
#         plt.ylabel('Score')
#         plt.title('[{}]'.format(folderName))
#         plt.legend(['Game score', 'min Game Score'], bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
#         plt.show()
#
#
# def addToInfo(info, key, value):
#     logger.info('Adding key [{}] to info'.format(key))
#     info[key] = value
#     return info
#
#
# def loadPlotInfo(jsonFullFname, info, keys):
#     infoModified = False
#
#     fname = 'info.log'
#     fnameFullPath = '{}/{}'.format(args.folderName, fname)
#     if os.path.exists(fnameFullPath):
#         with open(fnameFullPath, 'r') as f:
#             for key, func in keys.iteritems():
#                 if key not in info:
#                     info = addToInfo(info, key, func(f))
#                     infoModified = True
#
#             if infoModified is True:
#                 logger.info('Writing to [{}]'.format(jsonFullFname))
#                 with open(jsonFullFname, 'w') as fw:
#                     json.dump(info, fw)
#
#     return info
#
#
# def extractMinGameScoreFromFile(file):
#     v = None
#     for line in file.readlines():
#         idx = line.find('minGameScore')
#         if idx >= 0:
#             idx = line.find('[', idx)
#             if idx >= 0:
#                 idx2 = line.find(']', idx)
#                 v = eval(line[(idx + 1):idx2])
#                 break
#
#     return v
#
#
# def extractScoresFromFile(file):
#     scores = []
#     file.seek(0)
#
#     for line in file.readlines():
#         idx = line.find('score')
#         if idx >= 0:
#             idx = line.find('[', idx)
#             if idx >= 0:
#                 idx2 = line.find(']', idx)
#                 v = line[(idx + 1):idx2]
#                 scores.append(eval(v))
#
#     return scores
