import os
import json


def loadInfoFile(folderName, logger):
    jsonFname = 'info.json'
    jsonFullFname = '{}/{}'.format(folderName, jsonFname)
    info = {}

    if os.path.exists(jsonFullFname):
        with open(jsonFullFname, 'r') as f:
            logger.info('File [{}] exists, loading ...'.format(jsonFname))
            info = json.load(f)

    return info, jsonFullFname
