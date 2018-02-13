import numpy as np
from datetime import timedelta, datetime
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

    outputTitle = ['Room light1', 'Room light2', 'Room light3 (backdoor)', 'Kitchen light', 'Toilets light', 'Bathroom light', 'Living room light1',
                   'Living room light2', 'Hallway light', 'Entrance light', 'Boiler']

    inputTitle = ['Weekday', 'Hour', 'Minute']
    stateStartIdx = len(inputTitle)
    inputTitle.extend(outputTitle)

    # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
    nDays = 7
    weekdays = np.array([0, 1, 2, 3, 6])
    weekdays = weekdays / (nDays - 1)  # normalize to [0,1]

    weekend = np.array([4, 5])
    weekend = weekend / (nDays - 1)  # normalize to [0,1]

    def __init__(self):
        self.stateSize = len(self.inputTitle)
        self.actionSize = len(self.outputTitle)

    def minTimeUnit(self):
        return timedelta(minutes=1)

    def stateToDatetime(self, state):
        # 05/02/2018 is Monday which is (weekday == 0)
        return datetime(year=2018, month=2, day=5 + state[0], hour=state[1], minute=state[2])

    def buildNextState(self, nextDate, state, action):
        # update states (without date part)
        newState = (np.logical_xor(state[self.stateStartIdx:], action).astype(int))
        print(state[self.stateStartIdx:])
        print(action)
        print(newState)
        # create updated state (with date part)
        nextState = np.array([nextDate.weekday(), nextDate.hour, nextDate.minute])
        nextState = np.append(nextState, newState)

        return nextState

    # perform action
    def step(self, state, action):
        curDate = self.stateToDatetime(state)
        nextDate = curDate + self.minTimeUnit()
        nextState = self.buildNextState(nextDate, state, action)

        return nextState


G = WeekPolicy()

state = np.array([6, 20, 4], dtype=int)
print(state)
action = np.array([], dtype=int)
for i in range(len(G.outputTitle)):
    state = np.append(state, randint(0, 1))
    action = np.append(action, randint(0, 1))

nextState = G.step(state, action)
print(nextState)
