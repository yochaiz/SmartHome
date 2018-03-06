from keras.models import Model
from keras.layers import Dense, Input, add, Activation


class Critic:
    def __init__(self, seqLen, stateDim, actionDim):
        self.seqLen = seqLen
        self.stateDim = stateDim
        self.actionDim = actionDim

        self.model = self.buildModel()

    def buildModel(self):
        hidden1 = 512  # number of output units
        hidden2 = 256  # number of output units

        stateInput = Input(shape=(self.seqLen, self.stateDim))
        layer1 = Dense(hidden1, activation='relu')(stateInput)
        layer2State = Dense(hidden2, activation='linear')(layer1)

        actionInput = Input(shape=(self.actionDim,))
        layer2Action = Dense(hidden2, activation='linear')(actionInput)

        layer2 = add([layer2State, layer2Action])
        layer3 = Activation('relu')(layer2)
        layer4 = Dense(1, activation='linear')(layer3)

        model = Model(inputs=[stateInput, actionInput], outputs=layer4)

        return model
