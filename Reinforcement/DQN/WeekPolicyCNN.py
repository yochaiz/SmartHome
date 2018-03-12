from Reinforcement.DQN.WeekPolicyWithModel import WeekPolicyWithModel
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten


# Fully-connected based model
class WeekPolicyCNN(WeekPolicyWithModel):
    def __init__(self, fname, TAU, seqLen=1):
        super(WeekPolicyCNN, self).__init__(fname, TAU, seqLen)

    # build model to learn policy
    def buildModel(self):
        self.kernelSize = (1, 1)
        nChannels = 1
        # each state CONTAINS time prefix
        nFeatures = self.stateDevicesStartIdx + self.numOfDevices
        self.stateDim = (self.seqLen, nFeatures, nChannels)
        outputDim = pow(2, self.numOfDevices)  # possible actions

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Reshape(self.stateDim, input_shape=(self.seqLen, nFeatures)))
        model.add(Conv2D(256, kernel_size=self.kernelSize, activation='relu', input_shape=self.stateDim, data_format='channels_last'))
        # TODO: max pooling? dropout?
        model.add(Conv2D(128, kernel_size=self.kernelSize, activation='relu', data_format='channels_last'))
        model.add(Conv2D(64, kernel_size=self.kernelSize, activation='relu', data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(outputDim, activation='linear'))
        # model.add(Reshape((outputDim,), input_shape=self.stateDim))

        # set loss and optimizer
        model.compile(loss='mse', optimizer='adam')

        # print model architecture
        model.summary()

        return model
