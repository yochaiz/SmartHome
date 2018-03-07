from keras.models import Model
from keras.layers import Dense, Input, add, Activation
from keras.optimizers import Adam
from DeepNetwork import DeepNetwork
import tensorflow as tf


class Critic(DeepNetwork):
    def __init__(self, sess, seqLen, stateDim, actionDim, TAU, lr, nBackups):
        super(Critic, self).__init__(sess, seqLen, stateDim, actionDim, TAU, lr, nBackups)

        # set model optimization method (gradients calculation)
        self.action_grads = tf.gradients(self.models[self.trainModelKey].output, self.actionInput)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.stateInput: states,
            self.actionInput: actions
        })

    def buildModel(self, lr):
        hidden1 = 512  # number of output units
        hidden2 = 256  # number of output units

        self.stateInput = Input(shape=(self.seqLen, self.stateDim))
        layer1 = Dense(hidden1, activation='relu')(self.stateInput)
        layer2State = Dense(hidden2, activation='linear')(layer1)

        self.actionInput = Input(shape=(self.actionDim,))
        layer2Action = Dense(hidden2, activation='linear')(self.actionInput)

        layer2 = add([layer2State, layer2Action])
        layer3 = Activation('relu')(layer2)
        layer4 = Dense(1, activation='linear')(layer3)

        model = Model(inputs=[self.stateInput, self.actionInput], outputs=layer4)

        # compile model
        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam)

        return model
