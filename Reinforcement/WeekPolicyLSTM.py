import json
import numpy as np
from datetime import timedelta, datetime, time
from random import randint
from Policy import Policy
from keras.models import Sequential
from keras.layers import LSTM


class WeekPolicyLSTM(Policy):
    # This policy represents my typical week behavior as my house.
    # The policy is built from 7 days, 24 hours a day
    # List of objects I try to predicts:
    ## Lights:
    ### my room - 3
    ### kitchen - 1
    ### toilets - 1
    ### bathroom - 1
    ### living room - 2
    ### hallway - 1
    ### entrance door - 1
    #
    ## Boiler - 1 (we will assume winter time at the moment, i.e. every time I shower it requires boiler)
    #
    # Input structure:
    # [Weekday (0-6), Hour (0-23), Minute (0-60), Room light1, Room light2, Room light3,
    #  Kitchen light, Toilets light, Bathroom light, Living room light1, Living room light2,
    #  Hallway light, Entrance light, Boiler]
    #
    # Output structure:
    # [Room light1, Room light2, Room light3, Kitchen light, Toilets light,
    #  Bathroom light, Living room light1, Living room light2, Hallway light,
    #  Entrance light, Boiler]

    inputDateTitle = ['Weekday', 'Hour', 'Minute']
    stateDevicesStartIdx = len(inputDateTitle)

    # set values for state time normalization
    timeNormalizationValues = np.array([6, 24, 60], dtype=float)

    # Policy object structure:
    # Policy is a dictionary with key per device.
    ## Each key contains list of time dictionaries {'days':[] , 'times':[]}
    ### each time dictionary contains 2 keys:
    #### days: array of days the device should be ON.
    #### times: array of time interval during these days that the device should be ON.
    ##### each time interval in times is a tuple (startTime , endTime)

    # Input to model is like a sentence, i.e. a list of words
    # Each word represents a state WITH time prefix
    # Therefore, each word is a vector of length nDevices
    def __init__(self, fname, seqLen):
        super(WeekPolicyLSTM, self).__init__(fname, seqLen)

    @staticmethod
    def loadPolicyFromJSON(fname):
        with open(fname, 'r') as f:
            policy = json.load(f)

        WeekPolicyLSTM.validatePolicy(policy)
        return policy

    @staticmethod
    def validatePolicy(policy):
        nDays = len(policy["days"])
        for i in range(len(policy["Devices"])):
            device = policy[str(i)]
            daysArray = [[] for j in range(nDays)]

            for timeDict in device:
                if type(timeDict["days"]) is unicode:  # replace predefined array in JSON with actual array for future simplicity
                    timeDict["days"] = policy[timeDict["days"]]

                # sort timeDict by startTime
                timeDict["times"] = sorted(timeDict["times"], key=lambda t: datetime.strptime(t[0], policy["Time format"]))

                for day in timeDict["days"]:
                    daysArray[day].extend(timeDict["times"])

            # sort time ranges for easier compare
            for j in range(len(daysArray)):
                daysArray[j] = sorted(daysArray[j], key=lambda t: datetime.strptime(t[0], policy["Time format"]))

            for array in daysArray:
                for j in range(len(array) - 1):
                    tCur = array[j]
                    tNext = array[j + 1]
                    if tCur[1] > tNext[0]:  # endTime is bigger than next range startTime
                        raise ValueError(
                            'Validation failed for device [{}], ID:[{}], time ranges: [{}] - [{}]'.format(policy["Devices"][i], i, tCur, tNext))

    def minTimeUnit(self):
        return timedelta(minutes=1)

    # build model to learn policy
    def buildModel(self):
        # each state CONTAINS time prefix
        nFeatures = self.stateDevicesStartIdx + self.numOfDevices
        outputDim = pow(2, self.numOfDevices)  # possible actions

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(outputDim, activation='relu', input_shape=(self.seqLen, nFeatures), dropout=0.3, recurrent_dropout=0.3))

        # set loss and optimizer
        model.compile(loss='mse', optimizer='adam')

        # print model architecture
        model.summary()

        return model

    # Extracts date from given input, i.e. bottom row date
    def timePrefixToDate(self, input):
        # 05/02/2018 is Monday which is (weekday == 0)
        # it synchronizes between month day and weekday, i.e. same value for both
        return datetime(year=2018, month=2, day=5 + input[-1, 0], hour=input[-1, 1], minute=input[-1, 2])

    # Builds state at time nextDate based on current input and selected action
    def buildNextState(self, nextDate, input, action):
        # update states (without date part)
        newState = (np.logical_xor(input[-1, self.stateDevicesStartIdx:], action).astype(int))

        # create updated state (with date part)
        nextState = np.array([nextDate.weekday(), nextDate.hour, nextDate.minute], dtype=int)
        nextState = np.append(nextState, newState)

        newInput = np.copy(input)
        # remove current input top row
        newInput[:-1, :] = newInput[1:, :]
        # update new input bottom row to new date state
        newInput[-1, :] = nextState

        return newInput

    # Builds the expected state at time nextDate
    def buildExpectedState(self, nextDate):
        state = np.array([], dtype=int)
        for i in range(self.numOfDevices):
            device = self.policyJSON[str(i)]
            deviceState = 0
            for timeDict in device:
                if nextDate.weekday() in timeDict["days"]:
                    for t in timeDict["times"]:
                        if (nextDate.time() >= datetime.strptime(t[0], self.policyJSON["Time format"]).time()) \
                                and (nextDate.time() <= datetime.strptime(t[1], self.policyJSON["Time format"]).time()):
                            deviceState = 1
                            break

            state = np.append(state, deviceState)

        return state

    # Builds the expected input for model of given date
    def buildDateInput(self, stateTime):
        input = np.zeros((self.seqLen, self.stateDevicesStartIdx + self.numOfDevices), dtype=int)

        # build input according to seqLen
        for i in reversed(range(self.seqLen)):  # iterate backwards
            input[i, :self.stateDevicesStartIdx] = np.array([stateTime.weekday(), stateTime.hour, stateTime.minute], dtype=int)
            input[i, self.stateDevicesStartIdx:] = self.buildExpectedState(stateTime)
            stateTime -= self.minTimeUnit()

        return input

    # Calculates the reward based on the expected state at time nextDate compared to the actual state, nextState.
    # previous states that are part of the input to model are irrelevant to reward value
    def calculateReward(self, nextState, nextDate):
        expectedState = self.buildExpectedState(nextDate)
        # count how many devices we have predicted wrong
        wrongCounter = np.sum((np.logical_xor(nextState[self.stateDevicesStartIdx:], expectedState)))
        # count how many devices we have predicted correctly
        correctCounter = self.numOfDevices - wrongCounter

        # total reward is scaled in range [-1,1]
        reward = (correctCounter - wrongCounter) / float(self.numOfDevices)

        return reward

    # generate random time prefix for random state
    def generateRandomTimePrefix(self):
        # draw date
        day = randint(0, 6)
        hour = randint(0, 23)
        minute = randint(0, 59)

        # build state date prefix
        state = np.matrix([day, hour, minute], dtype=int)

        return state

    # normalize state vector before it goes through the model
    def normalizeStateForModelInput(self, state):
        input = state.astype(float)
        input[:, :len(self.timeNormalizationValues)] /= self.timeNormalizationValues
        return input

# seqLen = 60
# G = WeekPolicy("Week_policies/policy1.json", seqLen)
# input = np.ones((seqLen, 14))
# input = np.expand_dims(input, axis=-1)
# print(input.shape)
# output = G.model.predict(input)
# print(output.shape)
#
#
# state = np.array([6, 20, 4], dtype=int)
# action = np.array([], dtype=int)
# for i in range(len(G.policyJSON["Devices"])):
#     state = np.append(state, randint(0, 1))
#     action = np.append(action, randint(0, 1))
#
# print(state)
# nextState, reward = G.step(state, action)
# print(nextState)
# print(reward)
