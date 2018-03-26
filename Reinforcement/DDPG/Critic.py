from keras.models import Model, load_model
from keras.layers import Dense, Input, add, Activation, Reshape
from keras.optimizers import Adam
from DeepNetwork import DeepNetwork
import tensorflow as tf


class Critic(DeepNetwork):
    def __init__(self, sess, stateDim, actionDim, TAU, lr, nBackups):
        super(Critic, self).__init__(sess, stateDim, actionDim, TAU, lr, nBackups)

        # # load models from file
        # self.graph = tf.get_default_graph()
        # with self.graph.as_default():
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
        self.action_grads = tf.gradients(self.models[self.mainModelKey].output, self.actionInput)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.stateInput: states,
            self.actionInput: actions
        })[0]

    def buildModel(self, lr):
        graph = tf.get_default_graph()
        with graph.as_default():
            hidden1 = 512  # number of hidden layer 1 output units
            hidden2 = 256  # number of hidden layer 2 output units
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

        return model, graph
