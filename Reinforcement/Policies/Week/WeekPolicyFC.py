from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy
from keras.models import Sequential
from keras.layers import Dense, Reshape


# Fully-connected based model
class WeekPolicyFC(WeekPolicy):
    def __init__(self, fname, seqLen=1):
        super(WeekPolicyFC, self).__init__(fname, seqLen)

    # build model to learn policy
    def buildModel(self):
        self.stateDim = (self.seqLen, self.stateDevicesStartIdx + self.numOfDevices)
        outputDim = pow(2, self.numOfDevices)  # possible actions

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_shape=self.stateDim, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(outputDim, activation='linear'))
        model.add(Reshape((outputDim,), input_shape=self.stateDim))

        # set loss and optimizer
        model.compile(loss='mse', optimizer='adam')

        # print model architecture
        model.summary()

        return model
