from keras.models import Model
from keras.layers import Dense, Input
from DeepNetwork import DeepNetwork
import numpy as np
import tensorflow as tf


class Actor(DeepNetwork):
    def __init__(self, sess, policy, seqLen, stateDim, actionDim, TAU, lr, nBackups):
        super(Actor, self).__init__(sess, seqLen, stateDim, actionDim, TAU, lr, nBackups)

        # set model optimization method (gradients calculation)
        self.action_gradient = tf.placeholder(tf.float32, [None, actionDim])
        self.weights = self.models[self.trainModelKey].trainable_weights
        self.params_grad = tf.gradients(self.models[self.trainModelKey].output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

        # exploration params
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # exploration minimal rate
        self.epsilon_decay = 0.99  # 1 - 1E-3

        self.policy = policy

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.stateInput: states,
            self.action_gradient: action_grads
        })

    def buildModel(self, lr):
        self.stateInput = Input(shape=(self.seqLen, self.stateDim))
        h0 = Dense(512, activation='relu')(self.stateInput)
        # model.add(BatchNormalization())
        h1 = Dense(256, activation='relu')(h0)
        # model.add(BatchNormalization())
        V = Dense(self.actionDim, activation='sigmoid')(h1)
        # model.add(Reshape((self.actionDim,), input_shape=(self.seqLen, self.stateDim)))

        model = Model(input=self.stateInput, outputs=V)
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return self.policy.generateRandomAction(), 1

        # predict action from **train** network based on given state
        input = self.policy.normalizeStateForModelInput(state)
        input = np.expand_dims(input, axis=0)
        action = self.models[self.trainModelKey].predict(input)

        return action, 0
