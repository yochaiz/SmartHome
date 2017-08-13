from keras.models import load_model
import numpy as np
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def loadHdf5(fname):
    f = h5py.File(fname, 'r')
    x = f['default']
    print('data shape:%s' % str(x.shape))
    return x


fname = 'model-1.h5'
model = load_model(fname)

x = loadHdf5('x.hdf5')
i = 241672
print(x[i])
x = np.array(x)
y = loadHdf5('y.hdf5')
y = np.array(y)

print('Model prediction:')
print(model.predict(x[i].reshape((1, x[i].shape[0], x[i].shape[1]))))
print('Real label:')
print(y[i])
