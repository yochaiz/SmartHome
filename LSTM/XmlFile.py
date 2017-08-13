import xml.etree.ElementTree as Et


class XmlFile(object):
    def __init__(self, filePath):
        self.xml = Et.parse(filePath).getroot()
        self.pos = 0
        self.lastVal = 0
