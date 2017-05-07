from LightPoint import LightPoint
from RoomLights import RoomLights
from datetime import datetime
from ThermalProbe import ThermalProbe
from AuxChannel import AuxChannel
from Amplifier import Amplifier
from Microphone import Microphone

light = LightPoint('data\LightPoints\Devices.LightsAndAutomation.LightPoint.1.2.xml')
light.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# light.plotPtsRange(datetime(2016, 1, 23, 12, 30, 45), 7)

# room = RoomLights('1')
# room.plot(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# room.plot(datetime(2016, 2, 19, 21, 18, 00), datetime(2016, 2, 19, 21, 31, 00))

# probe = ThermalProbe('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml')
# probe.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))

# ir = AuxChannel('data/AuxChannel/Devices.Alarm.AuxChannel.4.xml')
# ir.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))

# ampf = Amplifier('data/Amplifier/Devices.Sound.Amplifier.1.1.xml')
# ampf.plotDateRange(datetime(2016, 2, 1, 19, 15, 00), datetime(2016, 2, 1, 19, 20, 00))

# mic = Microphone('data/Microphone/Devices.Microphone.NetworkMicrophone.1.xml.001.001.001.001.001')
# mic.plotDateRange(datetime(2016, 2, 16, 14, 49, 00), datetime(2016, 2, 16, 20, 20, 00))
