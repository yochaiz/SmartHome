from keras.models import Model
from keras.layers import Dense, Input, Reshape
from DeepNetwork import DeepNetwork
import numpy as np
from math import ceil
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


class Actor(DeepNetwork):
    # class of policy function pointers
    class PolicyFunctions:
        def __init__(self, idxToAction, generateRandomAction, normalizeState):
            self.idxToAction = idxToAction
            self.generateRandomAction = generateRandomAction
            self.normalizeState = normalizeState

    def __init__(self, sess, idxToAction, generateRandomAction, normalizeState, stateDim, actionDim, TAU, lr, nBackups):
        super(Actor, self).__init__(sess, stateDim, actionDim, TAU, lr, nBackups)

        # set model optimization method (gradients calculation)
        self.action_gradient = tf.placeholder(tf.float32, [None, actionDim])
        self.weights = self.models[self.mainModelKey].trainable_weights
        self.params_grad = tf.gradients(self.models[self.mainModelKey].output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

        # exploration params
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # exploration minimal rate
        self.epsilon_decay = 0.99  # 1 - 1E-3

        self.policy = self.PolicyFunctions(idxToAction, generateRandomAction, normalizeState)

        # init possible actions
        self.possibleActions, nActions = self.__buildPossibleActions()
        # number of knn neighbors to compare when converting continuous action to discrete action
        self.k = min(max(10, int(ceil(nActions * 0.1))), nActions)
        self.k = nActions
        # init knn object
        self.knn = NearestNeighbors(n_neighbors=self.k)
        # init knn object train set
        self.knn.fit(self.possibleActions)

    # Build matrix of possible actions given actionDim
    # for knn search of closest action for some continuous space action
    def __buildPossibleActions(self):
        nActions = pow(2, self.actionDim)
        actions = np.zeros((nActions, self.actionDim), dtype=int)
        for i in range(nActions):
            actions[i, :] = self.policy.idxToAction(i)

        return actions, nActions

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.stateInput: states,
            self.action_gradient: action_grads
        })

    def buildModel(self, lr):
        hidden1 = 2048
        hidden2 = 2048

        self.stateInput = Input(shape=self.stateDim)
        h0 = Dense(hidden1, activation='relu')(self.stateInput)
        # model.add(BatchNormalization())
        h1 = Dense(hidden2, activation='relu')(h0)
        # model.add(BatchNormalization())
        h2 = Dense(self.actionDim, activation='sigmoid')(h1)
        V = Reshape((self.actionDim,))(h2)

        model = Model(input=self.stateInput, outputs=V)
        return model

    def act(self, state, criticModel, optimalAction):
        isRandom = int(np.random.rand() <= self.epsilon)
        isInPool = 0

        if isRandom > 0:
            # The agent acts randomly
            action = self.policy.generateRandomAction()

        else:
            # predict action from **train** network based on given state
            input = self.policy.normalizeState(state)
            input = np.expand_dims(input, axis=0)
            action = self.models[self.mainModelKey].predict(input)
            # TODO: add action noise (Ornstein Uhlenbeck) ???
            # find IDs of closest valid (i.e. discrete, possible) actions
            # validActions = self.knn.kneighbors(action, return_distance=False)[0]
            distances, validActions = self.knn.kneighbors(action)
            maxPoolDist = distances[0].max()
            validActions = validActions[0]
            # convert IDs to the actions themselves
            validActions = self.possibleActions[validActions]
            # duplicate input as rows for Q-value prediction
            input = np.repeat(input, validActions.shape[0], axis=0)
            # calc Q-value for each valid action
            Qvalues = criticModel.predict([input, validActions])
            # choose highest Q-value action
            actionID = np.argmax(Qvalues)

            # check if optimalAction is in validActions pool
            dist = np.linalg.norm(action - optimalAction)
            isInPool = int(dist - maxPoolDist < 1E-5)

            # select optimal valid action
            action = validActions[actionID, :]

        isOptActionSelected = int(np.linalg.norm(action - optimalAction) < 1)

        return action, isRandom, isInPool, isOptActionSelected

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, (self.epsilon * self.epsilon_decay))
