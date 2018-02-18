import json
import numpy as np
from datetime import timedelta, datetime, time
from random import randint
from Policy import Policy


class WeekPolicy(Policy):
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

    # inputTitle.extend(outputTitle)

    # inputSize = len(inputTitle)
    # outputSize = len(outputTitle)

    # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
    # nDays = 7
    # weekdays = [0, 1, 2, 3, 6]
    # # weekdays = weekdays / (nDays - 1)  # normalize to [0,1]
    #
    # weekend = [4, 5]
    # # weekend = weekend / (nDays - 1)  # normalize to [0,1]

    # Policy object structure:
    # Policy is a dictionary with key per device.
    ## Each key contains list of time dictionaries {'days':[] , 'times':[]}
    ### each time dictionary contains 2 keys:
    #### days: array of days the device should be ON.
    #### times: array of time interval during these days that the device should be ON.
    ##### each time interval in times is a tuple (startTime , endTime)

    def __init__(self, fname):
        self.policyJSON = self.loadPolicyFromJSON(fname)
        self.nDevices = len(self.policyJSON["Devices"])

    @staticmethod
    def loadPolicyFromJSON(fname):
        with open(fname, 'r') as f:
            policy = json.load(f)

        WeekPolicy.validatePolicy(policy)
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

    def stateToDatetime(self, state):
        # 05/02/2018 is Monday which is (weekday == 0)
        # it synchronizes between month day and weekday, i.e. same value for both
        return datetime(year=2018, month=2, day=5 + state[0], hour=state[1], minute=state[2])

    def buildNextState(self, nextDate, state, action):
        # update states (without date part)
        newState = (np.logical_xor(state[self.stateDevicesStartIdx:], action).astype(int))

        # create updated state (with date part)
        nextState = np.array([nextDate.weekday(), nextDate.hour, nextDate.minute])
        nextState = np.append(nextState, newState)

        return nextState

    def buildExpectedState(self, nextDate):
        state = np.array([])
        for i in range(self.nDevices):
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

    def calculateReward(self, nextState, nextDate):
        expectedState = self.buildExpectedState(nextDate)
        # count how many devices we have predicted wrong
        wrongCounter = np.sum((np.logical_xor(nextState[self.stateDevicesStartIdx:], expectedState).astype(int)))
        # count how many devices we have predicted correctly
        correctCounter = self.nDevices - wrongCounter

        # total reward is scaled in range [-1,1]
        reward = (correctCounter - wrongCounter) / float(self.nDevices)

        return reward

    # perform action
    def step(self, state, action):
        curDate = self.stateToDatetime(state)
        nextDate = curDate + self.minTimeUnit()
        nextState = self.buildNextState(nextDate, state, action)

        # calculate reward
        reward = self.calculateReward(nextState, nextDate)

        return nextState, reward


G = WeekPolicy("Week_policies/policy1.json")

state = np.array([6, 20, 4], dtype=int)
action = np.array([], dtype=int)
for i in range(len(G.policyJSON["Devices"])):
    state = np.append(state, randint(0, 1))
    action = np.append(action, randint(0, 1))

print(state)
nextState, reward = G.step(state, action)
print(nextState)
print(reward)