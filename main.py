from LightPoint import LightPoint
from datetime import datetime
import matplotlib.dates as mdates

light = LightPoint('data\Devices.LightsAndAutomation.LightPoint.1.2.xml')
light.plot(datetime(2016, 1, 23, 12, 30, 45), datetime(2016, 1, 24, 12, 30, 45))
