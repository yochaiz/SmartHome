import os
from Features import Features
import pandas as pd
from sklearn import cluster
from sklearn import manifold
import matplotlib.pyplot as plt
from copy import copy

dstFileName = 'data/LP.csv'

# roomFolderName = 'data/LightPoints'
# if os.path.isdir(roomFolderName):
#     for filename in os.listdir(roomFolderName):
#         if filename.endswith(".xml"):
#             Features.writeToCSV(roomFolderName + '/' + filename, 'data/LP.csv')

X = pd.read_csv(dstFileName)
print(X.shape)

filenames = copy(X.iloc[:, 0])
for i in range(len(filenames)):
    filenames[i] = filenames[i][-7:-4]

# remove filename column
X = X.drop(X.columns[0], axis=1)
print(X.shape)

nClusters = 2
k_means = cluster.KMeans(n_clusters=nClusters, algorithm='full', max_iter=1000, tol=1e-8)
k_means.fit(X)
print('Performed [%d] iteration out of max_iter:[%d]' % (k_means.n_iter_, k_means.max_iter))

for i, label in enumerate(k_means.labels_):
    print('[%s] - Cluster:[%d]' % (filenames[i], label))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X)
print('Y shape:[%s]' % str(Y.shape))

fig = plt.figure()
colors = ['b', 'r']
for i in range(Y.shape[0]):
    plt.scatter(Y[i, 0], Y[i, 1], c=colors[k_means.labels_[i]], cmap=plt.cm.Spectral)
    plt.annotate(filenames[i], xy=(Y[i, 0], Y[i, 1]))

plt.title("t-SNE")
plt.axis('tight')
plt.xticks([])
plt.yticks([])
plt.show()
