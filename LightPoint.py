import xml.etree.ElementTree as Et
from datetime import datetime


class LightPoint(object):
    def __init__(self, filename):
        self.filename = filename
        self.root = Et.parse(filename).getroot()
        self.id = self.root.get('Id')

        for child in self.root:
            date = datetime.strptime(child.get('Time'), '%Y-%m-%d %H:%M:%S.%f')
            print(date.minute)
