from keras.models import load_model
import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf

# config = tf.ConfigProto()
# # limit memory precentage usage
# config.gpu_options.per_process_gpu_memory_fraction = 0.2

# set_session(tf.Session(config=config))


def loadHdf5(fname):
    f = h5py.File(fname, 'r')
    x = f['default']
    print('[{}] data shape:{}'.format(fname, x.shape))
    return x


def predictSingleSample(model, x, y, i):
    print('Model prediction:')
    print(model.predict(x[i].reshape((1, x[i].shape[0], x[i].shape[1]))))
    print('Real label:')
    print(y[i])


def thresholdPrediction(model, x, threshold, yPrediction):
    for i in range(len(x)):
        # for i in range(1):
        yPredicted = model.predict(x[i].reshape((1, x[i].shape[0], x[i].shape[1])))
        yPredicted = yPredicted[0]

        yPredicted[yPredicted >= threshold] = 1
        yPredicted[yPredicted < threshold] = 0

        yPrediction[i, :] = yPredicted

        if i % 1000 == 0:
            print(i)


def loadModelPredictions(fname, threshold, x, y):
    model = load_model(fname)
    yPredictionFileName = 'yPrediction-{}-{}'.format(threshold, fname)
    if not os.path.exists(yPredictionFileName):
        print('{} not found, creating it').format(yPredictionFileName)
        yPredictionFile = h5py.File(yPredictionFileName, 'w')
        yPrediction = yPredictionFile.create_dataset("default", y.shape)
        thresholdPrediction(model, x, threshold, yPrediction)
    else:
        print('{} found, loading it').format(yPredictionFileName)
        yPrediction = loadHdf5(yPredictionFileName)
        yPrediction = np.array(yPrediction)

    return yPrediction


class ROC:
    def __init__(self):
        self.truePositive = 0
        self.trueNegative = 0
        self.falsePositive = 0
        self.falseNegative = 0
        self.falseNegativeIdx = []
        self.falsePositiveIdx = []


def calcROC(y, yPrediction):
    if len(y) != len(yPrediction):
        raise ValueError('y & yPrediction have different lengths')

    res = []
    for i in range(y.shape[1]):
        res.append(ROC())

    for i in range(len(y)):
        # for i in range(10):
        if i % 10000 == 0:
            print(i)

        _y = y[i]
        _yPred = yPrediction[i]
        for j in range(len(_y)):
            if _y[j] == _yPred[j]:
                if _y[j] == 1:
                    res[j].truePositive += 1
                else:
                    res[j].trueNegative += 1
            else:
                if _y[j] == 1:
                    res[j].falseNegative += 1
                    res[j].falseNegativeIdx.append(i)
                else:
                    res[j].falsePositive += 1
                    res[j].falsePositiveIdx.append(i)

    return res


def loadROC(fname, threshold):
    rocFileName = 'roc-{}.json'.format(fname)
    x = None
    y = None
    yPrediction = None
    if not os.path.exists(rocFileName):
        x = loadHdf5('x.h5')
        x = np.array(x)
        y = loadHdf5('y.h5')
        y = np.array(y)
        yPrediction = loadModelPredictions(fname, threshold, x, y)

        print('{} not found, creating it').format(rocFileName)
        with open(rocFileName, 'w') as rocFile:
            res = calcROC(y, yPrediction)
            jRes = []
            for r in res:
                jRes.append(vars(r))

            json.dump(jRes, rocFile)
    else:
        print('{} found, loading it').format(rocFileName)
        with open(rocFileName, 'r') as rocFile:
            res = json.load(rocFile)

    return res, x, y, yPrediction


def plot(ax, startIdx, endIdx, y, yPrediction, colorLambdaFunc, label, startDate):
    colorLabels = [['T:[off]-P:[off]', 'T:[off]-P:[on]'], ['T:[on]-P:[off]', 'T:[on]-P:[on]']]
    colors = [['r', 'k'], ['b', 'g']]
    lastVal = 0
    val = 1
    width = []
    color = []
    leftValues = []
    legendMap = {}
    xticks = ax.get_xticks().tolist()
    xticklabels = ax.get_xticklabels()
    for idx in range(startIdx, endIdx + 1):
        width.append(val)
        yValue = int(y[idx])
        yPredValue = int(yPrediction[idx])
        color.append(colorLambdaFunc(colors, yValue, yPredValue))
        leftValues.append(lastVal)
        lastVal += val
        legendMap[colorLabels[yValue][yPredValue]] = idx - startIdx
        if (len(color) >= 2) and (color[-1] != color[-2]) and ((len(xticks) == 0) or ((idx - startIdx) - xticks[-1] >= 5)):
            xticks.append(idx - startIdx)
            newDate = startDate + timedelta(minutes=idx)
            xticklabels.append(newDate)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    height = 0.2
    plotData = [width, height, color, leftValues]

    yticks = ax.get_yticks().tolist()
    yticklabels = ax.get_yticklabels()
    pos = len(yticks)
    yticks.append(pos)
    pos -= (plotData[1] / 2.0)
    yticklabels.append(label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    h = ax.barh([pos] * len(plotData[0]), plotData[0], height=plotData[1], color=plotData[2], left=plotData[3],
                edgecolor='grey', linewidth=0.5)

    legend = ax.get_legend()
    labels = [] if legend is None else [x._text for x in legend.texts]
    handles = [] if legend is None else legend.legendHandles
    for key in legendMap.iterkeys():
        if key not in labels:
            handles.append(h[legendMap[key]])
            labels.append(key)
    # ax.legend(handles, labels, loc='upper right')
    ax.legend(handles, labels)


fname = 'model-1.h5'
threshold = 0.7
res, x, y, yPrediction = loadROC(fname, threshold)

for i, r in enumerate(res):
    str = '[{}]:\t'.format(i)
    for key in r:
        if not 'Idx' in key:
            str += '{}:[{}] \t\t'.format(key, r[key])
    print(str)

if y is None:
    y = loadHdf5('y.h5')
    y = np.array(y)

if yPrediction is None:
    if x is None:
        x = loadHdf5('x.h5')
        x = np.array(x)
    yPrediction = loadModelPredictions(fname, threshold, x, y)

rocColorLambdaFunc = lambda colors, yValue, yPredValue: colors[yValue][yPredValue]
trueColorLambdaFunc = lambda colors, yValue, yPredValue: colors[yValue][yValue]
predictColorLambdaFunc = lambda colors, yValue, yPredValue: colors[yPredValue][yPredValue]

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

intrvl = 100
dateFormat = '%Y-%m-%d %H:%M:%S'
startDate = datetime(2016, 1, 22, 22, 36)
for i, r in enumerate(res):
    for key in r:
        if ('Idx' in key) and (len(r[key]) > 0):
            i = 7
            idx = r[key][i]
            print(idx)
            curDate = startDate + timedelta(minutes=idx)
            plot(ax, idx - intrvl, idx + intrvl, y[:, i], yPrediction[:, i], rocColorLambdaFunc, 'ROC', startDate)
            plot(ax, idx - intrvl, idx + intrvl, y[:, i], yPrediction[:, i], trueColorLambdaFunc, 'True', startDate)
            plot(ax, idx - intrvl, idx + intrvl, y[:, i], yPrediction[:, i], predictColorLambdaFunc, 'Prediction', startDate)
            break
    break

ax.set_title('True label vs. Prediction')
fig.autofmt_xdate()
plt.show()

print('\nDone !')
