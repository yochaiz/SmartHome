import argparse
import os
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import logging
import matplotlib.pyplot as plt
import json


def plot(info):
    if info is not None:
        plt.plot(info['scores'], 'bo')
        plt.plot([info['minGameScore']] * len(info['scores']), 'r--')
        plt.xlabel('Game no.')
        plt.ylabel('Score')
        plt.legend(['Game score', 'min Game Score'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


def createJSON(jsonFullFname):
    fname = 'info.log'
    fnameFullPath = '{}/{}'.format(args.folderName, fname)
    info = None
    if os.path.exists(fnameFullPath):
        with open(fnameFullPath, 'r') as f:
            minGameScore = extractMinGameScoreFromFile(f)
            scores = extractScoresFromFile(f)
            info = {'scores': scores, 'minGameScore': minGameScore}
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
# parser.add_argument("gpuNum", type=int, help="GPU # to run on")
# parser.add_argument("--gpuFrac", type=float, default=0.3, help="GPU memory fraction")
args = parser.parse_args()

# # init GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuNum)
#
# # limit memory precentage usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = args.gpuFrac
# set_session(tf.Session(config=config))

# initialize logger
logging.basicConfig(level=logging.INFO, filename=args.folderName + '/plot.log')
logger = logging.getLogger(__name__)
logger.info('args:[{}]'.format(args))

jsonFname = 'info.json'
jsonFullFname = '{}/{}'.format(args.folderName, jsonFname)
info = None
if os.path.exists(jsonFullFname):
    with open(jsonFullFname, 'r') as f:
        logger.info('File [{}] exists, loading its data'.format(jsonFname))
        info = json.load(f)
else:
    logger.info('File [{}] does not exist, collecting its data'.format(jsonFname))
    info = createJSON(jsonFullFname)

plot(info)
