from Real import Real


class EnergyManagement(Real):
    nullTemp = 'NaN'

    def __init__(self, filename):
        super(EnergyManagement, self).__init__(filename, 'Energy [Watts]')
