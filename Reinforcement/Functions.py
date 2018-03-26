import os
import json
from datetime import datetime
import logging
import argparse
import sys
import re
import signal
from Reinforcement.Results import Results


def loadInfoFile(folderName, logger):
    jsonFname = 'info.json'
    jsonFullFname = '{}/{}'.format(folderName, jsonFname)
    info = {}

    if os.path.exists(jsonFullFname):
        with open(jsonFullFname, 'r') as f:
            info = json.load(f)
            if logger:
                logger.info('File [{}] exists, loading ...'.format(jsonFname))

    return info, jsonFullFname


# parse arguments
def parseArguments():
    parser = argparse.ArgumentParser(description='test model on dataset')
    parser.add_argument("gpuNum", type=int, help="GPU # to run on")
    parser.add_argument("--gpuFrac", type=float, default=0.3, help="GPU memory fraction")
    parser.add_argument("--settings", type=str, default='/home/yochaiz/SmartHome/Reinforcement/settings.json',
                        help="Settings JSON file")
    parser.add_argument("--desc", type=str, default=None, help="Experiment description")
    parser.add_argument("--k", type=int, choices=xrange(1, int(1E4) + 1), default=None,
                        help="Number of k nearest neighbors")
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

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    # limit memory precentage usage
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpuFrac
    sess = tf.Session(config=config)
    set_session(sess)
    return sess


# log info data
def logInfo(info, logger):
    for key in info:
        logger.info('{}:[{}]'.format(key, info[key]))


# log general & objects descriptions
def logDescriptions(logger, descArray):
    logger.info('===== DESCRIPTIONS =====')

    for objName, msgArray in descArray:
        for msg in msgArray:
            logger.info('[{}]: {}'.format(objName, msg))

    logger.info('===== ============ =====')


# save info data to JSON
def saveDataToJSON(info, jsonFullFname):
    with open(jsonFullFname, 'w') as f:
        json.dump(info, f)


# attach SIGTERM handler to program
def attachSIGTERMhandler(logger):
    # define terminate signal handler
    def terminateSignalHandler(signal, frame):
        if logger is not None:
            logger.info('_ _ _ Program terminated by user _ _ _')
        sys.exit(0)

    signal.signal(signal.SIGTERM, terminateSignalHandler)


def updateMaxTuple(newValue, g, curTuple):
    if newValue > curTuple[0]:
        curTuple = (newValue, [g])
    elif abs(newValue - curTuple[0]) < 1E-5:
        curTuple[1].append(g)

    return curTuple


# Collect score & loss from log file for future plots
def CollectResultsFromLog(folderName):
    logFname = '{}/info.log'.format(folderName)
    jsonFname = '{}/info.json'.format(folderName)

    # define regex
    pattern = re.compile('.*:episode: (\d+), score:\[(\d+\.\d+)\], loss:\[(\d+\.\d+)\]')

    # check files exist
    if not os.path.exists(jsonFname):
        print('path [{}] does not exist'.format(jsonFname))
        sys.exit()

    # load log file
    if not os.path.exists(logFname):
        print('path [{}] does not exist'.format(logFname))
        sys.exit()

    # load JSON file
    info, jsonFname = loadInfoFile(folderName, None)

    # init results object
    results = Results()

    # add results from log
    with open(logFname, 'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                results.score.append(int(round(float(m.group(2)))))
                results.loss.append(float(m.group(3)))

    # update info data in JSON, add results
    info['results'] = results.toJSON()
    saveDataToJSON(info, jsonFname)

    return info
