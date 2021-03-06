from OnOff import OnOff


# Infra-Red motion detector
class AuxChannel(OnOff):
    # nullColorDot = 'ko'
    # colorsDots = ['ro', 'go']

    nullColorBar = 'black'
    colorsBars = ['red', 'lightgreen']

    def __init__(self, filename):
        super(AuxChannel, self).__init__(filename, self.nullColorBar, self.colorsBars)

# def __init__(self, filename):
#     super(AuxChannel, self).__init__(filename, self.nullColorDot, self.colorsDots, self.nullColorBar,
#                                      self.colorsBars)
