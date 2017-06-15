import os
from Features import Features

roomFolderName = 'data/LightPoints'
dstFileName = 'data/LP.csv'
if os.path.isdir(roomFolderName):
    for filename in os.listdir(roomFolderName):
        if filename.endswith(".xml"):
            Features.writeToCSV(roomFolderName + '/' + filename, 'data/LP.csv')


