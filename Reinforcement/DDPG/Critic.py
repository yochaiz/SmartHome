from keras.models import Model
from keras.layers import Dense, Input, add, Activation, Reshape
from keras.optimizers import Adam
from DeepNetwork import DeepNetwork
import tensorflow as tf


class Critic(DeepNetwork):
    def __init__(self, sess, stateDim, actionDim, TAU, lr, nBackups):
        super(Critic, self).__init__(sess, stateDim, actionDim, TAU, lr, nBackups)

        # set model optimization method (gradients calculation)
        self.action_grads = tf.gradients(self.models[self.mainModelKey].output, self.actionInput)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.stateInput: states,
            self.actionInput: actions
        })[0]

    def buildModel(self, lr):
        hidden1 = 2048  # number of hidden layer 1 output units
        hidden2 = 2048  # number of hidden layer 2 output units
        nOutput = 1  # number of output layer units

        self.stateInput = Input(shape=self.stateDim)
        layer1 = Dense(hidden1, activation='relu')(self.stateInput)
        layer2State = Dense(hidden2, activation='linear')(layer1)

        self.actionInput = Input(shape=(self.actionDim,))
        layer2Action = Dense(hidden2, activation='linear')(self.actionInput)

        layer2 = add([layer2State, layer2Action])
        layer3 = Activation('relu')(layer2)
        layer4 = Dense(nOutput, activation='linear')(layer3)
        layer5 = Reshape((nOutput,))(layer4)

        model = Model(inputs=[self.stateInput, self.actionInput], outputs=layer5)

        # compile model
        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam)

        return model
