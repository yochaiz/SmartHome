from keras.models import Model, load_model
from keras.layers import Dense, Input, add, Activation, Reshape
from keras.optimizers import Adam
from DeepNetwork import DeepNetwork
import tensorflow as tf


class Critic(DeepNetwork):
    def __init__(self, sess, stateDim, actionDim, TAU, lr, nBackups):
        super(Critic, self).__init__(sess, stateDim, actionDim, TAU, lr, nBackups)

        # # load models from file
        #     self.models[self.mainModelKey] = load_model("results/D-22-3-H-14-19-41/Critic-main-model-0.h5")
        #     self.models[self.targetModelKey] = load_model("results/D-22-3-H-14-19-41/Critic-target-model-0.h5")
        #     # compile models
        #     adam = Adam(lr=lr)
        #     self.models[self.mainModelKey].compile(loss='mse', optimizer=adam)
        #     self.models[self.targetModelKey].compile(loss='mse', optimizer=adam)
        #
        #     self.stateInput = self.models[self.mainModelKey].input[0]
        #     self.actionInput = self.models[self.mainModelKey].input[1]

        # set model optimization method (gradients calculation)
        self.action_grads = tf.gradients(self.models[self.mainModelKey].output,
                                         self.actionInput)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.stateInput: states,
            self.actionInput: actions
        })[0]

    # ResNet architecture
    def buildModel(self, lr):
        self.description.append("ResNet architecture")
        hidden = 128
        nOutput = 1  # number of output layer units

        # define stateInput
        self.stateInput = Input(shape=self.stateDim)
        # project state input layer to hidden layers size
        projStateLayer = Dense(hidden)(self.stateInput)
        layer2State = self.buildBlock(hidden, self.stateInput, projStateLayer)

        self.actionInput = Input(shape=(self.actionDim,))
        # project action input layer to hidden layers size
        projActionLayer = Dense(hidden)(self.actionInput)

        layer2 = add([layer2State, projActionLayer])
        V = Activation('relu')(layer2)

        nLayers = 10
        for i in range(nLayers):
            V = self.buildBlock(hidden, V, V)

        # init output layer
        V = Dense(nOutput, activation='linear')(V)
        V = Reshape((nOutput,))(V)

        model = Model(inputs=[self.stateInput, self.actionInput], outputs=V)
        # compile model
        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam)

        return model

    # # deeper architecture
    # def buildModel(self, lr):
    #     self.description.append("Try deeper architecture")
    #     hidden = [256] * 2
    #     nOutput = 1  # number of output layer units
    #
    #     # define stateInput
    #     self.stateInput = Input(shape=self.stateDim)
    #     layer1 = Dense(hidden[0], activation='relu')(self.stateInput)
    #     layer2State = Dense(hidden[1], activation='linear')(layer1)
    #
    #     self.actionInput = Input(shape=(self.actionDim,))
    #     layer2Action = Dense(hidden[1], activation='linear')(self.actionInput)
    #
    #     layer2 = add([layer2State, layer2Action])
    #     layer3 = Activation('relu')(layer2)
    #
    #     nLayers = 3
    #     # init layers array
    #     h = [layer3]
    #     # add layers to array
    #     for i in range(nLayers):
    #         h.append(Dense(hidden[0], activation='relu')(h[-1]))
    #
    #     layer4 = Dense(nOutput, activation='linear')(h[-1])
    #     layer5 = Reshape((nOutput,))(layer4)
    #
    #     model = Model(inputs=[self.stateInput, self.actionInput], outputs=layer5)
    #     # compile model
    #     adam = Adam(lr=lr)
    #     model.compile(loss='mse', optimizer=adam)
    #
    #     return model

    # # Standard (paper) architecture
    # def buildModel(self, lr):
    #     self.description.append("Standard (paper) architecture")
    #     hidden = [512, 256]  # number of hidden layers output units
    #     nOutput = 1  # number of output layer units
    #
    #     # define stateInput
    #     self.stateInput = Input(shape=self.stateDim)
    #     layer1 = Dense(hidden[0], activation='relu')(self.stateInput)
    #     layer2State = Dense(hidden[1], activation='linear')(layer1)
    #
    #     self.actionInput = Input(shape=(self.actionDim,))
    #     layer2Action = Dense(hidden[1], activation='linear')(self.actionInput)
    #
    #     layer2 = add([layer2State, layer2Action])
    #     layer3 = Activation('relu')(layer2)
    #     layer4 = Dense(nOutput, activation='linear')(layer3)
    #     layer5 = Reshape((nOutput,))(layer4)
    #
    #     model = Model(inputs=[self.stateInput, self.actionInput], outputs=layer5)
    #     # compile model
    #     adam = Adam(lr=lr)
    #     model.compile(loss='mse', optimizer=adam)
    #
    #     return model
