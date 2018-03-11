from Reinforcement.Policies.Week.WeekPolicy import WeekPolicy
from keras.models import Sequential
from keras.layers import LSTM, Dense


# LSTM based model
class WeekPolicyLSTM(WeekPolicy):
    # Input to model is like a sentence, i.e. a list of words
    # Each word represents a state WITH time prefix
    # Therefore, each word is a vector of length nDevices
    def __init__(self, fname, seqLen):
        super(WeekPolicyLSTM, self).__init__(fname, seqLen)

    # build model to learn policy
    def buildModel(self):
        # each state CONTAINS time prefix
        nFeatures = self.stateDevicesStartIdx + self.numOfDevices
        self.stateDim = (self.seqLen, nFeatures)
        outputDim = pow(2, self.numOfDevices)  # possible actions

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(256, activation='relu', input_shape=self.stateDim, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        model.add(LSTM(128, activation='relu', dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(outputDim, activation='linear'))

        # set loss and optimizer
        model.compile(loss='mse', optimizer='adam')

        # print model architecture
        model.summary()

        return model
