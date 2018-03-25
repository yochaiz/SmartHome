from keras.models import Model
from keras.layers import Dense, Input, Reshape
from keras.optimizers import Adam
from DeepNetwork import DeepNetwork
import numpy as np
from math import ceil
from threading import Thread
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


class Actor(DeepNetwork):
    # class of policy function pointers
    class PolicyFunctions:
        def __init__(self, idxToAction, generateRandomAction, normalizeState):
            self.idxToAction = idxToAction
            self.generateRandomAction = generateRandomAction
            self.normalizeState = normalizeState

    def __init__(self, sess, idxToAction, generateRandomAction, normalizeState, stateDim, actionDim, TAU, lr, k, nBackups):
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
        self.possibleActions, self.nActions = self.__buildPossibleActions()
        # number of knn neighbors to compare when converting continuous action to discrete action
        self.k = min(max(10, int(ceil(self.nActions * 0.1))), self.nActions) if k is None else min(k, self.nActions)
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
        graph = tf.get_default_graph()
        with graph.as_default():
            hidden1 = 512
            hidden2 = 256

            self.stateInput = Input(shape=self.stateDim)
            h0 = Dense(hidden1, activation='relu')(self.stateInput)
            # model.add(BatchNormalization())
            h1 = Dense(hidden2, activation='relu')(h0)
            # model.add(BatchNormalization())
            h2 = Dense(self.actionDim, activation='sigmoid')(h1)
            V = Reshape((self.actionDim,))(h2)

            model = Model(input=self.stateInput, outputs=V)
            # compile model
            adam = Adam(lr=lr)
            model.compile(loss='mse', optimizer=adam)

        return model, graph

    def __optimalActionPerState(self, state, criticModel, criticModelGraph, validActions, discreteAction, i):
        # criticModelGraph is required for tf model context in different thread
        with criticModelGraph.as_default():
            # duplicate input as number of actions for Q-value prediction
            input = np.expand_dims(state, axis=0)
            input = np.repeat(input, self.k, axis=0)
            # calc Q-value for each valid action
            Qvalues = criticModel.predict([input, validActions])
            # choose highest Q-value action
            actionID = np.argmax(Qvalues)
            # select optimal valid action
            discreteAction[i] = validActions[actionID, :]

    # state is the vector from Policy object, AFTER normalization
    def wolpertingerAction(self, state, actorModel, criticModel, criticModelGraph):
        contAction = actorModel.predict(state)
        # TODO: add action noise (Ornstein Uhlenbeck) ???
        # find IDs of closest valid (i.e. discrete, possible) actions
        # validActions = self.knn.kneighbors(action, return_distance=False)[0]
        distances, validActions = self.knn.kneighbors(contAction)
        # find max pool distance for each action in contAction
        maxPoolDist = distances.max(axis=-1)
        # convert IDs to the actions themselves
        validActions = self.possibleActions[validActions]
        # evaluate Qvalues for each state
        nSamples = state.shape[0]
        # init selected discrete action for each state
        discreteAction = np.zeros((nSamples, self.actionDim), dtype=int)
        # init threadsPool
        threadsPool = []
        for i in range(nSamples):
            t = Thread(target=self.__optimalActionPerState,
                       args=(state[i], criticModel, criticModelGraph, validActions[i], discreteAction, i))
            threadsPool.append(t)

        # start all threads
        for t in threadsPool:
            t.start()

        # wait for all threads to complete
        for t in threadsPool:
            t.join()

        return discreteAction, contAction, maxPoolDist

    def act(self, state, criticModel, criticModelGraph, optimalAction):
        isRandom = int(np.random.rand() <= self.epsilon)
        isInPool = 0

        if isRandom > 0:
            # The agent acts randomly
            discreteAction = self.policy.generateRandomAction()

        else:
            # predict discrete action from **MAIN** network based on given state
            state = self.policy.normalizeState(state)
            state = np.expand_dims(state, axis=0)
            discreteAction, contAction, maxPoolDist = self.wolpertingerAction(state, self.models[self.mainModelKey], criticModel,
                                                                              criticModelGraph)
            discreteAction = discreteAction[0]
            contAction = contAction[0]
            maxPoolDist = maxPoolDist[0]

            # check if optimalAction is in validActions pool
            dist = np.linalg.norm(contAction - optimalAction)
            isInPool = int(dist - maxPoolDist < 1E-5)

        isOptActionSelected = int(np.linalg.norm(discreteAction - optimalAction) < 1)

        return discreteAction, isRandom, isInPool, isOptActionSelected

    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, (self.epsilon * self.epsilon_decay))
