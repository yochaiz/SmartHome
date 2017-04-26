from LightPoint import LightPoint
from RoomLights import RoomLights
from datetime import datetime
from ThermalProbe import ThermalProbe

light = LightPoint('data\LightPoints\Devices.LightsAndAutomation.LightPoint.1.2.xml')
light.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
# light.plotPtsRange(datetime(2016, 1, 23, 12, 30, 45), 7)

# room = RoomLights('1')
# room.plot(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))

# probe = ThermalProbe('data/ThermalProbe/Devices.ClimateControl.ThermalProbe.1.xml')
# probe.plotDateRange(datetime(2016, 2, 15, 8, 00, 45), datetime(2016, 2, 15, 21, 30, 53))
