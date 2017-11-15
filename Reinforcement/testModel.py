import os
from keras.models import load_model
import numpy as np
import argparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import logging
import json
from Functions import loadInfoFile


def infoToLogger(logger, info):
    logger.info('Results:')
    logger.info('======================')
    logger.info('Total: Mistakes:[{}] \t\t Ratio:[{}]'.format(info['totalAcc']['Quantity'], info['totalAcc']['Ratio']))
    logger.info('\n')

    for h, data in info['accuracyPerHour'].iteritems():
        logger.info('Hour:[{}] \t\t Mistakes:[{}] \t\t Ratio:[{}]'.format(h, data['Quantity'], data['Ratio']))


# parse arguments
parser = argparse.ArgumentParser(description='test model on dataset')
parser.add_argument("folderName", type=str, help="Folder name where the model for testing is located")
parser.add_argument("gpuNum", type=int, help="GPU # to run on")
parser.add_argument("--gpuFrac", type=float, default=0.3, help="GPU memory fraction")
args = parser.parse_args()

# init GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuNum)

# limit memory precentage usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpuFrac
set_session(tf.Session(config=config))

# initialize logger
logging.basicConfig(level=logging.INFO, filename=args.folderName + '/test.log')
logger = logging.getLogger(__name__)
logger.info('args:[{}]'.format(args))

fname = '{}/model.h5'.format(args.folderName)
if not os.path.exists(fname):
    logger.info('Path [{}] does not exist'.format(fname))
    exit(0)

model = load_model(fname)
model.summary(print_fn=lambda x: logger.info(x))
model.summary()

actions = ['Do not change status', 'Change light status']
nActions = len(actions)
states = []
for v in range(nActions):
    states += [[0.0, 0.0, v]] * 60
states = np.asarray(states)

nHours = 24
mins = np.arange(0, 1, 1 / 60.0)

# update minutes column
pos = 0
for _ in range(nActions):
    states[pos:(pos + len(mins)), 1] = mins
    pos += len(mins)

# info = {'totalAcc': {}, 'accuracyPerHour': {}}
info, jsonFullFname = loadInfoFile(args.folderName, logger)
newKeys = ['accuracyPerHour', 'totalAcc']
for key in newKeys:
    info[key] = {}

totalCounter = 0
# predict each hour
for h in range(nHours):
    info['accuracyPerHour'][h] = {}
    infoElement = info['accuracyPerHour'][h]

    states[:, 0] = h / float(nHours)
    predictions = model.predict(states)
    predictions = np.argmax(predictions, axis=1)  # maximun in each row, i.e. for each prediction
    predictions = abs(predictions - states[:, 2])  # calculate diff between prediction to real label

    localCounter = np.sum(predictions)
    infoElement['Quantity'] = localCounter
    infoElement['Ratio'] = localCounter / float(len(mins))

    totalCounter += localCounter

info['totalAcc']['Quantity'] = totalCounter
info['totalAcc']['Ratio'] = totalCounter / (float(len(mins)) * nHours)

with open(jsonFullFname, 'w') as f:
    json.dump(info, f)

infoToLogger(logger, info)
