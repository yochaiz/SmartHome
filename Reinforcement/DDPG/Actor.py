from keras.models import Sequential
from keras.layers import Dense
from DeepNetwork import DeepNetwork


class Actor(DeepNetwork):
    def __init__(self, seqLen, stateDim, actionDim, TAU, nBackups):
        super(Actor, self).__init__(seqLen, stateDim, actionDim, TAU, nBackups)

    def buildModel(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.seqLen, self.stateDim), activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(self.actionDim, activation='sigmoid'))
        # model.add(Reshape((self.actionDim,), input_shape=(self.seqLen, self.stateDim)))

        return model


