from AuxChannel import AuxChannel
from ThermalProbe import ThermalProbe
import os


class Room:
    deviceMap = {'AuxChannel': AuxChannel, 'ThermalProbe': ThermalProbe}

    def __init__(self, roomFolderName):
        self.devices = {}
        for key in self.deviceMap.keys():
            folderPath = roomFolderName + '/' + key
            if os.path.isdir(folderPath):
                self.devices[key] = []
                for filename in os.listdir(folderPath):
                    if filename.endswith(".xml"):
                        self.devices[key].append(self.deviceMap[key](folderPath + '/' + filename))



