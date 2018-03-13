import os
import json
from datetime import datetime
import logging
import argparse


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
    parser.add_argument("--settings", type=str, default='/home/yochaiz/SmartHome/Reinforcement/settings.json', help="Settings JSON file")
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


# save info data to JSON
def saveDataToJSON(info, jsonFullFname):
    with open(jsonFullFname, 'w') as f:
        json.dump(info, f)


def updateMaxTuple(newValue, g, curTuple):
    if newValue > curTuple[0]:
        curTuple = (newValue, [g])
    elif abs(newValue - curTuple[0]) < 1E-5:
        curTuple[1].append(g)

    return curTuple
