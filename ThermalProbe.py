from Real import Real


class ThermalProbe(Real):
    nullTemp = 'NaN'

    def __init__(self, filename):
        super(ThermalProbe, self).__init__(filename,'Temperature [Celsius]')
