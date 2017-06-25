from OnOff import OnOff


class LightPoint(OnOff):
    # nullColorDot = 'ro'
    # colorsDots = ['ko', 'yo']

    nullColorBar = 'red'
    colorsBars = ['black', 'yellow']

    def __init__(self, filename):
        super(LightPoint, self).__init__(filename, self.nullColorBar, self.colorsBars)

# def __init__(self, filename):
#     super(LightPoint, self).__init__(filename, self.nullColorDot, self.colorsDots, self.nullColorBar,
#                                      self.colorsBars)
