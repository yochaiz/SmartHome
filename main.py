from LightPoint import LightPoint
from RoomLights import RoomLights
from datetime import datetime, timedelta
from ThermalProbe import ThermalProbe
from AuxChannel import AuxChannel
from Amplifier import Amplifier
from Microphone import Microphone
from EnergyManagement import EnergyManagement
from Room import Room
import matplotlib.pyplot as plt

#
# fig, ax = plt.subplots()
# ax.barh([0.5, 1], [2, 3], height=0.2, color=['red', 'blue'], left=[2, 0], edgecolor='grey', linewidth=0.5)
# bgcolor = 0.95
# ax.set_axis_bgcolor((bgcolor, bgcolor, bgcolor))
# plt.show()

# room = Room('data/LivingRoom')
# room.plotDateRange(datetime(2016, 2, 16, 14, 30, 0), datetime(2016, 2, 16, 23, 0, 0))
# room.plotDateRange(datetime(2016, 2, 19, 21, 00, 0), datetime(2016, 2, 21, 13, 30, 0))
# room.plotRepeatDateRange(datetime(2016, 1, 23, 7, 0, 0), datetime(2016, 1, 23, 23, 0, 0), 'LightPoints', 2,
#                          timedelta(days=7))

room2 = Room('data/MasterBedroom')
room2.plotDateRange(datetime(2016, 2, 19, 21, 00, 0), datetime(2016, 2, 21, 13, 30, 0), timedelta(minutes=1).seconds)

# probe = ThermalProbe('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml')
# probe.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))

plt.show()

# en = EnergyManagement('data/Energy/Devices.EnergyManagement.EnergyManagementCentral.1.xml')
# en.plotDateRange(datetime(2016, 2, 19, 21, 00, 0), datetime(2016, 2, 21, 13, 30, 0), timedelta(minutes=1).seconds)

# light = LightPoint('data\LightPoints\Devices.LightsAndAutomation.LightPoint.1.2.xml')
# light.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# startDate, seqLen = light.findSequence(20, timedelta(minutes=2))
# print(startDate, seqLen)
# light.showBarsPlot(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# light.plotPtsRange(datetime(2016, 1, 23, 12, 30, 45), 7)

# room = RoomLights('1')
# room.plot(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# room.plot(datetime(2016, 2, 19, 21, 18, 00), datetime(2016, 2, 19, 21, 31, 00))

# ir = AuxChannel('data/AuxChannel/Devices.Alarm.AuxChannel.1.4.xml')
# # ir.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# ir.plotBars(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))

# ampf = Amplifier('data/Amplifier/Devices.Sound.Amplifier.1.1.xml')
# ampf.plotDateRange(datetime(2016, 2, 1, 19, 15, 00), datetime(2016, 2, 1, 19, 20, 00))

# mic = Microphone('data/Microphone/Devices.Microphone.NetworkMicrophone.1.xml.001.001.001.001.001')
# mic.plotDateRange(datetime(2016, 2, 16, 14, 49, 00), datetime(2016, 2, 16, 16, 20, 00))
